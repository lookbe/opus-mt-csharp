using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Text.Json;
using static MyApp.MarianTokenizerShim;

namespace MyApp
{
    // --- ModelInput (Data structure for ONNX inputs) ---
    public class ModelInput
    {
        public List<long> InputIds { get; set; } = new List<long>();
        public List<long> AttentionMask { get; set; } = new List<long>();
        public long[] InputIdsShape { get; set; } = { 1, 0 };
        public long[] MaskShape { get; set; } = { 1, 0 };
    }

    // --- MarianTokenizerShim (Tokenizer logic and helper classes) ---
    public class MarianTokenizerShim
    {
        private const string SPIECE_UNDERLINE = "\u2581";

        public SentencePieceTokenizer.SentencePieceTokenizer SpmSource;
        public SentencePieceTokenizer.SentencePieceTokenizer SpmTarget;

        public Dictionary<string, int> Encoder { get; } = new Dictionary<string, int>();
        public Dictionary<int, string> Decoder { get; } = new Dictionary<int, string>();

        public string UnkToken { get; }
        public string EosToken { get; }
        public string PadToken { get; }

        public int UnkTokenId { get; }
        public int EosTokenId { get; }
        public int PadTokenId { get; }

        public List<string> AllSpecialTokens { get; }
        public int ModelMaxLength { get; }

        public class BiasDefinition
        {
            public List<int> SequenceIds { get; set; }
            public float BiasValue { get; set; }
        }

        // --- SequenceBiasLogitsProcessor ---
        public class SequenceBiasLogitsProcessor
        {
            private Dictionary<List<int>, float> sequenceBias;
            private float[] length1Bias;
            private bool preparedBiasVariables = false;

            public SequenceBiasLogitsProcessor(List<BiasDefinition> sequenceBiasList)
            {
                this.sequenceBias = ConvertListArgumentsIntoDictionary(sequenceBiasList);
            }

            private Dictionary<List<int>, float> ConvertListArgumentsIntoDictionary(List<BiasDefinition> sequenceBiasList)
            {
                var convertedBias = new Dictionary<List<int>, float>(new ListComparer());
                foreach (var item in sequenceBiasList)
                {
                    convertedBias[item.SequenceIds] = item.BiasValue;
                }
                return convertedBias;
            }

            private void PrepareBiasVariables(float[,] scores)
            {
                int vocabularySize = scores.GetLength(1);

                var invalidBiases = new List<int>();
                foreach (var sequenceIds in sequenceBias.Keys)
                {
                    foreach (int tokenId in sequenceIds)
                    {
                        if (tokenId >= vocabularySize)
                        {
                            invalidBiases.Add(tokenId);
                        }
                    }
                }

                if (invalidBiases.Any())
                {
                    throw new ArgumentOutOfRangeException(
                        $"The model vocabulary size is {vocabularySize}, but the following tokens were being biased: " +
                        $"{string.Join(", ", invalidBiases.Distinct())}"
                    );
                }

                length1Bias = new float[vocabularySize];
                foreach (var kvp in sequenceBias)
                {
                    if (kvp.Key.Count == 1)
                    {
                        length1Bias[kvp.Key[0]] = kvp.Value;
                    }
                }

                preparedBiasVariables = true;
            }

            /// <summary>
            /// Applies the sequence bias to the token scores (logits).
            /// </summary>
            public float[,] Invoke(int[][] inputIds, float[,] scores)
            {
                int batchSize = scores.GetLength(0);
                int vocabularySize = scores.GetLength(1);
                int currentSequenceLength = inputIds[0].Length;

                if (!preparedBiasVariables)
                {
                    PrepareBiasVariables(scores);
                }

                var bias = new float[batchSize, vocabularySize];

                // 1. Include the bias from length = 1
                for (int i = 0; i < batchSize; i++)
                {
                    for (int j = 0; j < vocabularySize; j++)
                    {
                        bias[i, j] += length1Bias[j];
                    }
                }

                // 2. Include the bias from length > 1
                foreach (var kvp in sequenceBias)
                {
                    var sequenceIds = kvp.Key;
                    float sequenceBiasValue = kvp.Value;

                    if (sequenceIds.Count == 1)
                    {
                        continue;
                    }

                    int prefixLength = sequenceIds.Count - 1;

                    if (prefixLength >= currentSequenceLength)
                    {
                        continue;
                    }

                    int lastToken = sequenceIds.Last();

                    for (int i = 0; i < batchSize; i++)
                    {
                        bool prefixMatches = true;
                        int startTokenIndex = currentSequenceLength - prefixLength;

                        for (int j = 0; j < prefixLength; j++)
                        {
                            if (inputIds[i][startTokenIndex + j] != sequenceIds[j])
                            {
                                prefixMatches = false;
                                break;
                            }
                        }

                        if (prefixMatches)
                        {
                            bias[i, lastToken] += sequenceBiasValue;
                        }
                    }
                }

                // 3. Apply the bias to the scores and return the processed scores
                var scoresProcessed = new float[batchSize, vocabularySize];
                for (int i = 0; i < batchSize; i++)
                {
                    for (int j = 0; j < vocabularySize; j++)
                    {
                        scoresProcessed[i, j] = scores[i, j] + bias[i, j];
                    }
                }

                return scoresProcessed;
            }

            // Custom IEqualityComparer to allow using List<int> as a Dictionary key
            private class ListComparer : IEqualityComparer<List<int>>
            {
                public bool Equals(List<int> x, List<int> y)
                {
                    if (x == null || y == null) return x == y;
                    return x.SequenceEqual(y);
                }

                public int GetHashCode(List<int> obj)
                {
                    if (obj == null) return 0;
                    int hash = 19;
                    foreach (var item in obj)
                    {
                        hash = hash * 31 + item.GetHashCode();
                    }
                    return hash;
                }
            }
        }

        // --- Constructor and Tokenizer Methods ---
        public MarianTokenizerShim(
            string sourceSpm,
            string targetSpm,
            string vocab,
            string unkTok,
            string eosTok,
            string padTok,
            int maxLen)
        {
            UnkToken = unkTok;
            EosToken = eosTok;
            PadToken = padTok;
            ModelMaxLength = maxLen;

            // 1. Load SentencePiece models
            try
            {
                SpmSource = new SentencePieceTokenizer.SentencePieceTokenizer(sourceSpm);
                SpmTarget = new SentencePieceTokenizer.SentencePieceTokenizer(targetSpm);
            }
            catch (FileNotFoundException ex)
            {
                throw new Exception("Failed to load SentencePiece model(s).", ex);
            }

            // 2. Load Vocabulary (token-to-ID mapping)
            try
            {
                string jsonString = File.ReadAllText(vocab);
                var jsonDocument = JsonDocument.Parse(jsonString);

                foreach (var property in jsonDocument.RootElement.EnumerateObject())
                {
                    Encoder[property.Name] = property.Value.GetInt32();
                }
            }
            catch (Exception ex) when (ex is FileNotFoundException || ex is JsonException)
            {
                throw new Exception("Failed to load or parse vocab.json file.", ex);
            }

            // 3. Special Tokens setup and ID access
            if (!Encoder.ContainsKey(UnkToken)) throw new Exception("UNK token not in vocab");
            if (!Encoder.ContainsKey(EosToken)) throw new Exception("EOS token not in vocab");
            if (!Encoder.ContainsKey(PadToken)) throw new Exception("PAD token not in vocab");

            UnkTokenId = Encoder[UnkToken];
            EosTokenId = Encoder[EosToken];
            PadTokenId = Encoder[PadToken];

            AllSpecialTokens = new List<string> { UnkToken, EosToken, PadToken };

            foreach (var pair in Encoder)
            {
                Decoder[pair.Value] = pair.Key;
            }
        }

        private long ConvertTokenToId(string token)
        {
            return Encoder.TryGetValue(token, out int id) ? id : UnkTokenId;
        }

        public List<string> Tokenize(string text)
        {
            var pieces = SpmSource.EncodeToStrings(text.AsSpan());
            return pieces.ToList();
        }

        public List<long> ConvertTokensToIds(List<string> tokens)
        {
            return tokens.Select(ConvertTokenToId).ToList();
        }

        public List<long> BuildInputsWithSpecialTokens(List<long> tokenIds)
        {
            var inputs = new List<long>(tokenIds);
            inputs.Add(EosTokenId);
            return inputs;
        }

        public ModelInput Call(string text, bool truncation = true)
        {
            List<string> tokens = Tokenize(text);
            List<long> tokenIds = ConvertTokensToIds(tokens);
            List<long> inputIdsInt = BuildInputsWithSpecialTokens(tokenIds);

            if (truncation && inputIdsInt.Count > ModelMaxLength)
            {
                inputIdsInt.RemoveRange(ModelMaxLength - 1, inputIdsInt.Count - (ModelMaxLength - 1));
                inputIdsInt[inputIdsInt.Count - 1] = EosTokenId;
            }

            int seqLen = inputIdsInt.Count;

            var result = new ModelInput();
            result.InputIds.AddRange(inputIdsInt);
            result.AttentionMask.AddRange(Enumerable.Repeat<long>(1, seqLen));
            result.InputIdsShape = new long[] { 1, seqLen };
            result.MaskShape = new long[] { 1, seqLen };

            return result;
        }

        public string ConvertTokensToString(List<string> tokens)
        {
            var currentSubTokens = new List<int>();
            string outString = "";

            foreach (var token in tokens)
            {
                bool isSpecial = AllSpecialTokens.Contains(token);
                if (isSpecial)
                {
                    if (currentSubTokens.Count > 0)
                    {
                        string decodedPiece = SpmTarget.Decode(currentSubTokens.ToArray());
                        outString += decodedPiece + token + " ";
                    }

                    currentSubTokens.Clear();
                }
                else
                {
                    // Convert piece → id
                    var ids = SpmTarget.EncodeToIds(token.AsSpan());
                    currentSubTokens.Add(ids[0]);
                }
            }

            if (currentSubTokens.Count > 0)
            {
                string final = SpmTarget.Decode(currentSubTokens.ToArray());
                outString += final;
            }

            // Cleanup
            outString = outString.Replace(SPIECE_UNDERLINE, " ");
            outString = outString.TrimEnd(' ', '\n', '\r', '\t');

            return outString;
        }

        public string Decode(List<long> tokenIds, bool skipSpecialTokens = true)
        {
            var tokens = new List<string>();
            foreach (int id in tokenIds)
            {
                if (!Decoder.TryGetValue(id, out string token))
                {
                    token = UnkToken;
                }

                bool isSpecial = AllSpecialTokens.Contains(token);

                if (skipSpecialTokens && isSpecial)
                {
                    continue;
                }

                tokens.Add(token);
            }

            return ConvertTokensToString(tokens);
        }
    }

    // --- MarianTokenizeProgram (Main inference loop) ---
    internal class MarianTokenizeProgram
    {
        private const int NumLayers = 6;
        private const int NumPastTensorsPerLayerFull = 4;
        private const int NumPastTensorsPerLayerDecoderOnly = 2;

        private static (long[] InputIds, long[] AttentionMask) CreateInitialDecoderInput(MarianTokenizerShim tokenizer, int batchSize)
        {
            long startTokenId = tokenizer.PadTokenId;
            var decoderInputIds = new long[batchSize];
            Array.Fill(decoderInputIds, startTokenId);
            var decoderAttentionMask = new long[batchSize];
            Array.Fill(decoderAttentionMask, 1);
            return (decoderInputIds, decoderAttentionMask);
        }

        public static string RunMarianOnnxInference(
            InferenceSession encoderSession,
            InferenceSession decoderSession,
            InferenceSession decoderWithPastSession,
            MarianTokenizerShim tokenizer,
            SequenceBiasLogitsProcessor logitsProcessor,
            string inputText,
            int maxLength)
        {
            var inputs = tokenizer.Call(inputText);
            var inputIds = inputs.InputIds.ToArray();
            var attentionMask = inputs.AttentionMask.ToArray();
            var batchSize = 1;

            // 1. ENCODER PASS
            var encoderInputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input_ids", new DenseTensor<long>(inputIds, new int[] { 1, inputIds.Length })),
                NamedOnnxValue.CreateFromTensor("attention_mask", new DenseTensor<long>(attentionMask, new int[] { 1, attentionMask.Length }))
            };

            using (var encoderOutputs = encoderSession.Run(encoderInputs))
            {
                var encoderHiddenStatesTensor = encoderOutputs.First().AsTensor<float>();
                var encoderHiddenStates = encoderHiddenStatesTensor.AsEnumerable<float>().ToArray();
                var dimensions = encoderHiddenStatesTensor.Dimensions.ToArray();
                int sequenceLength = dimensions[1];
                int hiddenSize = dimensions[2];

                // 2. DECODER SETUP
                var initialDecoderInputs = CreateInitialDecoderInput(tokenizer, batchSize);
                long[] currentDecoderInputIds = initialDecoderInputs.InputIds;
                List<(Tensor<float> DecK, Tensor<float> DecV, Tensor<float> EncK, Tensor<float> EncV)> pastKeyValues = null;

                List<long> fullDecodedIdsList = new List<long>(initialDecoderInputs.InputIds);
                List<long> decodedTokenIds = new List<long>();

                // 3. GENERATION LOOP (Autoregressive decoding)
                for (int i = 0; i < maxLength; i++)
                {
                    var decoderInputs = new List<NamedOnnxValue>();
                    InferenceSession session;

                    if (pastKeyValues == null)
                    {
                        session = decoderSession;
                        decoderInputs.Add(NamedOnnxValue.CreateFromTensor("input_ids", new DenseTensor<long>(currentDecoderInputIds, new int[] { 1, currentDecoderInputIds.Length })));
                        var encHiddenStatesShape = new int[] { batchSize, sequenceLength, hiddenSize };
                        decoderInputs.Add(NamedOnnxValue.CreateFromTensor("encoder_hidden_states", new DenseTensor<float>(encoderHiddenStates, encHiddenStatesShape)));
                        decoderInputs.Add(NamedOnnxValue.CreateFromTensor("encoder_attention_mask", new DenseTensor<long>(attentionMask, new int[] { 1, attentionMask.Length })));
                    }
                    else
                    {
                        session = decoderWithPastSession;
                        decoderInputs.Add(NamedOnnxValue.CreateFromTensor("input_ids", new DenseTensor<long>(currentDecoderInputIds, new int[] { 1, currentDecoderInputIds.Length })));
                        decoderInputs.Add(NamedOnnxValue.CreateFromTensor("encoder_attention_mask", new DenseTensor<long>(attentionMask, new int[] { 1, attentionMask.Length })));

                        for (int layerIdx = 0; layerIdx < pastKeyValues.Count; layerIdx++)
                        {
                            var layerPast = pastKeyValues[layerIdx];
                            decoderInputs.Add(NamedOnnxValue.CreateFromTensor($"past_key_values.{layerIdx}.decoder.key", layerPast.DecK));
                            decoderInputs.Add(NamedOnnxValue.CreateFromTensor($"past_key_values.{layerIdx}.decoder.value", layerPast.DecV));
                            decoderInputs.Add(NamedOnnxValue.CreateFromTensor($"past_key_values.{layerIdx}.encoder.key", layerPast.EncK));
                            decoderInputs.Add(NamedOnnxValue.CreateFromTensor($"past_key_values.{layerIdx}.encoder.value", layerPast.EncV));
                        }
                    }

                    using (var decoderOutputs = session.Run(decoderInputs))
                    {
                        var newPastKeyValues = new List<(Tensor<float> DecK, Tensor<float> DecV, Tensor<float> EncK, Tensor<float> EncV)>();
                        var logitsTensor = decoderOutputs.First().AsTensor<float>();

                        if (pastKeyValues == null)
                        {
                            for (int layerIdx = 0; layerIdx < NumLayers; layerIdx++)
                            {
                                int startIndex = 1 + layerIdx * NumPastTensorsPerLayerFull;

                                // IMPORTANT FIX: Clone the tensors to prevent reference-disposal issues
                                var decK = decoderOutputs[startIndex].AsTensor<float>().Clone();
                                var decV = decoderOutputs[startIndex + 1].AsTensor<float>().Clone();
                                var encK = decoderOutputs[startIndex + 2].AsTensor<float>().Clone();
                                var encV = decoderOutputs[startIndex + 3].AsTensor<float>().Clone();

                                newPastKeyValues.Add((decK, decV, encK, encV));
                            }
                        }
                        else
                        {
                            for (int layerIdx = 0; layerIdx < NumLayers; layerIdx++)
                            {
                                int startIndex = 1 + layerIdx * NumPastTensorsPerLayerDecoderOnly;

                                // IMPORTANT FIX: Clone the tensors to prevent reference-disposal issues
                                var newDecK = decoderOutputs[startIndex].AsTensor<float>().Clone();
                                var newDecV = decoderOutputs[startIndex + 1].AsTensor<float>().Clone();

                                var layerPast = pastKeyValues[layerIdx];
                                var staticEncK = layerPast.EncK;
                                var staticEncV = layerPast.EncV;
                                newPastKeyValues.Add((newDecK, newDecV, staticEncK, staticEncV));

                                if (layerPast.DecK is IDisposable decKDisposable)
                                {
                                    decKDisposable.Dispose();
                                }
                                if (layerPast.DecV is IDisposable decVDisposable)
                                {
                                    decVDisposable.Dispose();
                                }
                            }

                            for (int layerIdx = 0; layerIdx < pastKeyValues.Count; layerIdx++)
                            {
                                if (pastKeyValues[layerIdx].EncK is IDisposable encKDisposable)
                                {
                                    encKDisposable.Dispose();
                                }
                                if (pastKeyValues[layerIdx].EncV is IDisposable encVDisposable)
                                {
                                    encVDisposable.Dispose();
                                }
                            }
                        }

                        // Assign the new past key values for the next iteration
                        pastKeyValues = newPastKeyValues;

                        int vocabSize = logitsTensor.Dimensions[2];
                        int currentOutputSequenceLength = logitsTensor.Dimensions[1];

                        // Extract next_token_logits (logits[:, -1, :])
                        float[,] nextTokenLogits = new float[batchSize, vocabSize];
                        for (int v = 0; v < vocabSize; v++)
                        {
                            nextTokenLogits[0, v] = logitsTensor[0, currentOutputSequenceLength - 1, v];
                        }

                        // Prepare for LogitsProcessor
                        int[][] fullDecodedIdsSoFar = new int[batchSize][];
                        fullDecodedIdsSoFar[0] = fullDecodedIdsList.Select(id => (int)id).ToArray();

                        // Apply Logits Processor
                        float[,] processedLogits = nextTokenLogits;
                        if (logitsProcessor != null)
                        {
                            processedLogits = logitsProcessor.Invoke(fullDecodedIdsSoFar, nextTokenLogits);
                        }

                        // Greedy Search (Argmax)
                        long nextTokenId = -1;
                        float maxLogit = float.NegativeInfinity;

                        for (int v = 0; v < vocabSize; v++)
                        {
                            float logit = processedLogits[0, v];
                            if (logit > maxLogit)
                            {
                                maxLogit = logit;
                                nextTokenId = v;
                            }
                        }

                        if (nextTokenId == tokenizer.EosTokenId)
                        {
                            break;
                        }

                        // Update for Next Step
                        decodedTokenIds.Add(nextTokenId);
                        fullDecodedIdsList.Add(nextTokenId);
                        currentDecoderInputIds = new long[] { nextTokenId };
                    }
                }

                return tokenizer.Decode(decodedTokenIds, skipSpecialTokens: true);
            }
        }

        static void Main(string[] args)
        {
            var sentence = "halo apa kabar";
            const int MAX_LENGTH = 50;
            const string LOCAL_DIR = @"D:/ai/onnx/opus-mt/id-to-en";

            Console.WriteLine($"Source: {sentence}");
            Console.WriteLine("------------------------------");

            try
            {
                var tokenizer = new MarianTokenizerShim(
                    Path.Combine(LOCAL_DIR, "source.spm"),
                    Path.Combine(LOCAL_DIR, "target.spm"),
                    Path.Combine(LOCAL_DIR, "vocab.json"),
                    "<unk>", "</s>", "<pad>", MAX_LENGTH
                );

                var SEQUENCE_BIAS_CONFIG = new List<BiasDefinition>
                {
                    new BiasDefinition
                    {
                        SequenceIds = new List<int> { tokenizer.PadTokenId },
                        BiasValue = float.NegativeInfinity
                    }
                };

                SequenceBiasLogitsProcessor biasProcessor = new SequenceBiasLogitsProcessor(SEQUENCE_BIAS_CONFIG);

                using var encoder_session = new InferenceSession(Path.Combine(LOCAL_DIR, "encoder_model.onnx"));
                using var decoder_session = new InferenceSession(Path.Combine(LOCAL_DIR, "decoder_model.onnx"));
                using var decoder_with_past_session = new InferenceSession(Path.Combine(LOCAL_DIR, "decoder_with_past_model.onnx"));

                var translated_text = RunMarianOnnxInference(
                    encoder_session,
                    decoder_session,
                    decoder_with_past_session,
                    tokenizer,
                    biasProcessor,
                    sentence,
                    MAX_LENGTH
                );

                Console.WriteLine($"Translation: **{translated_text}**");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"\nError during execution: {ex.Message}");
            }
        }
    }
}