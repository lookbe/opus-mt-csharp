using BERTTokenizers;
using Microsoft.ML.Data;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using static MyApp.MarianTokenizerShim;

namespace MyApp // Note: actual namespace depends on the project name.
{
    // Represents the output required by the neural network model
    public class ModelInput
    {
        // The token IDs (padded/truncated sequence)
        public List<long> InputIds { get; set; } = new List<long>();

        // The attention mask (1 for valid tokens, 0 for padding - simple implementation here is all 1s)
        public List<long> AttentionMask { get; set; } = new List<long>();

        // Shape for the input_ids tensor: {BatchSize, SequenceLength}
        public long[] InputIdsShape { get; set; } = { 1, 0 };

        // Shape for the attention_mask tensor: {BatchSize, SequenceLength}
        public long[] MaskShape { get; set; } = { 1, 0 };
    }

    public class MarianTokenizerShim
    {
        // Constant for the SentencePiece underline character (U+2581)
        private const string SPIECE_UNDERLINE = "\u2581";

        // SentencePiece processors (must be initialized with an actual implementation)
        public SentencePieceTokenizer.SentencePieceTokenizer SpmSource;
        public SentencePieceTokenizer.SentencePieceTokenizer SpmTarget;

        // Vocabulary maps
        public Dictionary<string, int> Encoder { get; } = new Dictionary<string, int>();
        public Dictionary<int, string> Decoder { get; } = new Dictionary<int, string>();

        // Special Tokens
        public string UnkToken { get; }
        public string EosToken { get; }
        public string PadToken { get; }

        public int UnkTokenId { get; }
        public int EosTokenId { get; }
        public int PadTokenId { get; }

        public List<string> AllSpecialTokens { get; }
        public int ModelMaxLength { get; }

        // Define a type alias for a list of tokens and its associated bias,
        // which matches the Python input format: [List<int>, float]
        // In C#, we use 'Tuple<List<int>, float>' or a simple custom class/struct.
        public class BiasDefinition
        {
            public List<int> SequenceIds { get; set; }
            public float BiasValue { get; set; }
        }

        public class SequenceBiasLogitsProcessor
        {
            private Dictionary<List<int>, float> sequenceBias;
            private float[] length1Bias;
            private bool preparedBiasVariables = false;

            // The constructor accepts the sequence bias in a structured C# format
            public SequenceBiasLogitsProcessor(List<BiasDefinition> sequenceBiasList)
            {
                this.sequenceBias = ConvertListArgumentsIntoDictionary(sequenceBiasList);
            }

            /// <summary>
            /// Converts the list of BiasDefinition objects into a dictionary for easier lookup.
            /// The list of tokens is used as the key.
            /// </summary>
            private Dictionary<List<int>, float> ConvertListArgumentsIntoDictionary(List<BiasDefinition> sequenceBiasList)
            {
                var convertedBias = new Dictionary<List<int>, float>(new ListComparer());
                foreach (var item in sequenceBiasList)
                {
                    // Use a custom equality comparer for List<int> keys
                    convertedBias[item.SequenceIds] = item.BiasValue;
                }
                return convertedBias;
            }

            /// <summary>
            /// Precomputes the bias for length=1 sequences and validates tokens.
            /// </summary>
            private void PrepareBiasVariables(float[,] scores)
            {
                // scores shape: [batch_size, vocab_size]
                int vocabularySize = scores.GetLength(1);

                // 1. Check biased tokens out of bounds
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

                // 2. Precompute the bias for length 1 sequences.
                length1Bias = new float[vocabularySize]; // Initialized to 0.0f
                foreach (var kvp in sequenceBias)
                {
                    if (kvp.Key.Count == 1)
                    {
                        // The bias is applied to the single token ID
                        length1Bias[kvp.Key[0]] = kvp.Value;
                    }
                }

                preparedBiasVariables = true;
            }

            /// <summary>
            /// Applies the sequence bias to the token scores (logits).
            /// </summary>
            /// <param name="inputIds">The current sequence IDs: [batch_size, current_sequence_length]</param>
            /// <param name="scores">The current logits: [batch_size, vocab_size]</param>
            /// <returns>The processed scores.</returns>
            public float[,] Invoke(int[][] inputIds, float[,] scores)
            {
                // input_ids shape: [batch_size, current_sequence_length]
                // scores shape: [batch_size, vocab_size]
                int batchSize = scores.GetLength(0);
                int vocabularySize = scores.GetLength(1);
                int currentSequenceLength = inputIds[0].Length;

                // 1 - Prepares the bias tensors.
                if (!preparedBiasVariables)
                {
                    PrepareBiasVariables(scores);
                }

                // 2 - Prepare a copy of scores to add the bias to (or an empty bias array)
                // Since C# doesn't have NumPy's `zeros_like` and we want an additive bias,
                // we'll start with a bias array initialized to zero.
                var bias = new float[batchSize, vocabularySize];

                // 3 - Include the bias from length = 1 (applied to all batches)
                for (int i = 0; i < batchSize; i++)
                {
                    for (int j = 0; j < vocabularySize; j++)
                    {
                        bias[i, j] += length1Bias[j];
                    }
                }

                // 4 - Include the bias from length > 1, after determining which sequences may be completed.
                foreach (var kvp in sequenceBias)
                {
                    var sequenceIds = kvp.Key;
                    float sequenceBiasValue = kvp.Value;

                    if (sequenceIds.Count == 1)
                    {
                        // Already applied in step 3
                        continue;
                    }

                    int prefixLength = sequenceIds.Count - 1;

                    // Skip if the sequence prefix is longer than the current context
                    if (prefixLength > currentSequenceLength) // Note: Python used '>=' but in C# we use '>' as the current token is not included
                    {
                        continue;
                    }

                    int lastToken = sequenceIds.Last(); // The token to be biased

                    // Check if the generated sequence ends with the required prefix for each batch item
                    for (int i = 0; i < batchSize; i++)
                    {
                        bool prefixMatches = true;

                        // Compare the last 'prefixLength' tokens of the input_ids
                        for (int j = 0; j < prefixLength; j++)
                        {
                            // inputIds[i][inputIds[i].Length - prefixLength + j]
                            // inputIds[i].Length - prefixLength is the start index of the prefix in the current sequence

                            int currentInputTokenIndex = currentSequenceLength - prefixLength + j;
                            int requiredPrefixToken = sequenceIds[j];

                            if (inputIds[i][currentInputTokenIndex] != requiredPrefixToken)
                            {
                                prefixMatches = false;
                                break;
                            }
                        }

                        // If the prefix matches, apply the bias to the 'lastToken' logit for this batch item
                        if (prefixMatches)
                        {
                            bias[i, lastToken] += sequenceBiasValue;
                        }
                    }
                }

                // 5 - apply the bias to the scores and return the processed scores
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
                    if (x == null || y == null)
                    {
                        return x == y;
                    }
                    return x.SequenceEqual(y);
                }

                public int GetHashCode(List<int> obj)
                {
                    if (obj == null) return 0;
                    // Simple hash combining approach
                    int hash = 19;
                    foreach (var item in obj)
                    {
                        hash = hash * 31 + item.GetHashCode();
                    }
                    return hash;
                }
            }
        }

        // --- Constructor ---
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

            // 2. Load Vocabulary (token-to-ID mapping) using System.Text.Json
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
            if (!Encoder.ContainsKey(UnkToken))
                throw new Exception("UNK token not in vocab");
            if (!Encoder.ContainsKey(EosToken))
                throw new Exception("EOS token not in vocab");
            if (!Encoder.ContainsKey(PadToken))
                throw new Exception("PAD token not in vocab");

            UnkTokenId = Encoder[UnkToken];
            EosTokenId = Encoder[EosToken];
            PadTokenId = Encoder[PadToken];

            AllSpecialTokens = new List<string> { UnkToken, EosToken, PadToken };

            // Populate the reverse map for decoding
            foreach (var pair in Encoder)
            {
                Decoder[pair.Value] = pair.Key;
            }
        }

        // --- Private Methods ---

        private long ConvertTokenToId(string token)
        {
            return Encoder.TryGetValue(token, out int id) ? id : UnkTokenId;
        }

        public List<string> Tokenize(string text)
        {
            var pieces = SpmSource.EncodeToStrings(text.AsSpan());
            return pieces.ToList();
        }

        // --- Public Methods (Core Logic) ---

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

        // C++ operator() logic implemented here
        public ModelInput Call(string text, bool truncation = true)
        {
            // 1. Tokenize and Convert to IDs
            List<string> tokens = Tokenize(text);
            List<long> tokenIds = ConvertTokensToIds(tokens);

            // 2. Add special tokens (append EOS)
            List<long> inputIdsInt = BuildInputsWithSpecialTokens(tokenIds);

            // 3. Truncation
            if (truncation && inputIdsInt.Count > ModelMaxLength)
            {
                // Truncate to MaxLength - 1, then ensure the last token is EOS
                inputIdsInt.RemoveRange(ModelMaxLength - 1, inputIdsInt.Count - (ModelMaxLength - 1));
                inputIdsInt[inputIdsInt.Count - 1] = EosTokenId;
            }

            // 4. Padding/Attention Mask (simple implementation for one sentence)
            int seqLen = inputIdsInt.Count;

            var result = new ModelInput();
            result.InputIds.AddRange(inputIdsInt);
            // Mask is 1 for all valid tokens (no padding yet)
            result.AttentionMask.AddRange(Enumerable.Repeat<long>(1, seqLen));

            // Finalize shapes for ONNX (Batch=1, SeqLen)
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
                // Find token string from ID
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
    internal class MarianTokenizeProgram
    {
        private const int NumLayers = 6;
        private const int NumPastTensorsPerLayerFull = 4; // Dec K, Dec V, Enc K, Enc V
        private const int NumPastTensorsPerLayerDecoderOnly = 2; // New Dec K, New Dec V

        // Helper to create the initial decoder input
        private static (long[] InputIds, long[] AttentionMask) CreateInitialDecoderInput(MarianTokenizerShim tokenizer, int batchSize)
        {
            long startTokenId = tokenizer.PadTokenId;

            // Initial input_ids shape: (batch_size, 1)
            var decoderInputIds = new long[batchSize];
            for (int i = 0; i < batchSize; i++)
            {
                decoderInputIds[i] = startTokenId;
            }

            // Initial attention_mask shape: (batch_size, 1)
            var decoderAttentionMask = new long[batchSize];
            for (int i = 0; i < batchSize; i++)
            {
                decoderAttentionMask[i] = 1;
            }

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
            var batchSize = 1; // Marian is typically run with batchSize=1

            // --- 1. ENCODER PASS ---
            var encoderInputs = new List<NamedOnnxValue>
    {
        NamedOnnxValue.CreateFromTensor(
            "input_ids",
            new DenseTensor<long>(
                inputIds,
                new int[] { 1, inputIds.Length }
            )
        ),
        NamedOnnxValue.CreateFromTensor(
            "attention_mask",
            new DenseTensor<long>(
                attentionMask,
                new int[] { 1, attentionMask.Length }
            )
        )
    };

            using (var encoderOutputs = encoderSession.Run(encoderInputs))
            {
                var encoderHiddenStatesTensor = encoderOutputs.First().AsTensor<float>();
                var encoderHiddenStates = encoderHiddenStatesTensor.AsEnumerable<float>().ToArray();

                var dimensions = encoderHiddenStatesTensor.Dimensions.ToArray();
                int sequenceLength = dimensions[1];
                int hiddenSize = dimensions[2];

                // --- 2. DECODER SETUP ---
                var initialDecoderInputs = CreateInitialDecoderInput(tokenizer, batchSize);
                long[] currentDecoderInputIds = initialDecoderInputs.InputIds;

                List<(Tensor<float> DecK, Tensor<float> DecV, Tensor<float> EncK, Tensor<float> EncV)> pastKeyValues = null;

                // This list tracks ALL tokens generated so far, required for the LogitsProcessor.
                // It starts with the initial token (e.g., the decoder_start_token_id).
                List<long> fullDecodedIdsList = new List<long>(initialDecoderInputs.InputIds);

                List<long> decodedTokenIds = new List<long>(); // Stores only the *newly* generated tokens (excluding the start token)

                // --- 3. GENERATION LOOP ---
                for (int i = 0; i < maxLength; i++)
                {
                    // ... (ONNX setup for decoderInputs and session selection remains the same) ...
                    var decoderInputs = new List<NamedOnnxValue>();
                    InferenceSession session;

                    if (pastKeyValues == null)
                    {
                        session = decoderSession;

                        decoderInputs.Add(NamedOnnxValue.CreateFromTensor("input_ids", new DenseTensor<long>(currentDecoderInputIds, new int[] { 1, currentDecoderInputIds.Length })));
                        var encHiddenStatesShape = new int[] { batchSize, sequenceLength, hiddenSize };
                        decoderInputs.Add(NamedOnnxValue.CreateFromTensor(
                            "encoder_hidden_states",
                            new DenseTensor<float>(encoderHiddenStates, encHiddenStatesShape)
                        ));
                        decoderInputs.Add(NamedOnnxValue.CreateFromTensor("encoder_attention_mask", new DenseTensor<long>(attentionMask, new int[] { 1, attentionMask.Length })));
                    }
                    else
                    {
                        session = decoderWithPastSession;

                        // When using past, input_ids only contains the *last* generated token
                        decoderInputs.Add(NamedOnnxValue.CreateFromTensor("input_ids", new DenseTensor<long>(currentDecoderInputIds, new int[] { 1, currentDecoderInputIds.Length })));
                        decoderInputs.Add(NamedOnnxValue.CreateFromTensor("encoder_attention_mask", new DenseTensor<long>(attentionMask, new int[] { 1, attentionMask.Length })));

                        for (int layerIdx = 0; layerIdx < pastKeyValues.Count; layerIdx++)
                        {
                            var layerPast = pastKeyValues[layerIdx];
                            // ... (Add past key/value tensors as before) ...
                            decoderInputs.Add(NamedOnnxValue.CreateFromTensor($"past_key_values.{layerIdx}.decoder.key", layerPast.DecK));
                            decoderInputs.Add(NamedOnnxValue.CreateFromTensor($"past_key_values.{layerIdx}.decoder.value", layerPast.DecV));
                            decoderInputs.Add(NamedOnnxValue.CreateFromTensor($"past_key_values.{layerIdx}.encoder.key", layerPast.EncK));
                            decoderInputs.Add(NamedOnnxValue.CreateFromTensor($"past_key_values.{layerIdx}.encoder.value", layerPast.EncV));
                        }
                    }
                    // ... (End of ONNX setup) ...

                    using (var decoderOutputs = session.Run(decoderInputs))
                    {
                        var newPastKeyValues = new List<(Tensor<float> DecK, Tensor<float> DecV, Tensor<float> EncK, Tensor<float> EncV)>();
                        var logitsTensor = decoderOutputs.First().AsTensor<float>();

                        // ... (Logic to update newPastKeyValues remains the same) ...
                        if (pastKeyValues == null)
                        {
                            // Full past extraction logic
                            for (int layerIdx = 0; layerIdx < NumLayers; layerIdx++)
                            {
                                int startIndex = 1 + layerIdx * NumPastTensorsPerLayerFull;
                                var decK = decoderOutputs[startIndex].AsTensor<float>();
                                var decV = decoderOutputs[startIndex + 1].AsTensor<float>();
                                var encK = decoderOutputs[startIndex + 2].AsTensor<float>();
                                var encV = decoderOutputs[startIndex + 3].AsTensor<float>();
                                newPastKeyValues.Add((decK, decV, encK, encV));
                            }
                        }
                        else
                        {
                            // Decoder-only past extraction logic (encoder past is static)
                            for (int layerIdx = 0; layerIdx < NumLayers; layerIdx++)
                            {
                                int startIndex = 1 + layerIdx * NumPastTensorsPerLayerDecoderOnly;
                                var newDecK = decoderOutputs[startIndex].AsTensor<float>();
                                var newDecV = decoderOutputs[startIndex + 1].AsTensor<float>();
                                var layerPast = pastKeyValues[layerIdx];
                                var staticEncK = layerPast.EncK;
                                var staticEncV = layerPast.EncV;
                                newPastKeyValues.Add((newDecK, newDecV, staticEncK, staticEncV));
                            }
                        }
                        pastKeyValues = newPastKeyValues;

                        int vocabSize = logitsTensor.Dimensions[2];
                        // sequenceLength is the length of the new sequence output by the decoder. 
                        // When using past, this is 1. When not using past (first step), it's input length + 1.
                        int currentOutputSequenceLength = logitsTensor.Dimensions[1];

                        // Extract next_token_logits (logits[:, -1, :]) - the logits for the last token generated
                        float[,] nextTokenLogits = new float[batchSize, vocabSize];
                        for (int v = 0; v < vocabSize; v++)
                        {
                            nextTokenLogits[0, v] = logitsTensor[0, currentOutputSequenceLength - 1, v];
                        }

                        // --- FIX: Derive fullDecodedIdsSoFar from the list ---
                        // The processor expects a jagged array (int[][]) of the *already generated* tokens.
                        // Note: The LogitsProcessor expects `int` tokens, but the generation loop uses `long`. 
                        // We cast here, assuming token IDs fit within an int.
                        int[][] fullDecodedIdsSoFar = new int[batchSize][];
                        fullDecodedIdsSoFar[0] = fullDecodedIdsList.Select(id => (int)id).ToArray();
                        // --- END FIX ---

                        // --- NEW LOGIC: Apply Logits Processor ---
                        float[,] processedLogits = nextTokenLogits;

                        if (logitsProcessor != null)
                        {
                            processedLogits = logitsProcessor.Invoke(fullDecodedIdsSoFar, nextTokenLogits);
                        }

                        // Apply argmax (Greedy Search) to the processed logits (which is [1, vocab_size])
                        long nextTokenId = -1;
                        float maxLogit = float.MinValue;

                        for (int v = 0; v < vocabSize; v++)
                        {
                            float logit = processedLogits[0, v];
                            if (logit > maxLogit)
                            {
                                maxLogit = logit;
                                nextTokenId = v;
                            }
                        }

                        // Check for End-of-Sentence token
                        if (nextTokenId == tokenizer.EosTokenId)
                        {
                            break; // Exit the token generation loop
                        }

                        // 9. Update for Next Step
                        decodedTokenIds.Add(nextTokenId);

                        // --- FIX: Update the running list of all tokens generated ---
                        fullDecodedIdsList.Add(nextTokenId);
                        // --- END FIX ---

                        // Input for the next decoder step is only the single new token
                        currentDecoderInputIds = new long[] { nextTokenId };
                    }
                }

                var finalOutput = tokenizer.Decode(decodedTokenIds, skipSpecialTokens: true);
                return finalOutput;

            }
        }

        static void Main(string[] args)
            {
                var sentence = "halo apa kabar";
                Console.WriteLine(sentence);

                var tokenizer = new MarianTokenizerShim(
                    @"D:\tools\opus-mt-onnx\my_local_tokenizer_files\source.spm",
                    @"D:\tools\opus-mt-onnx\my_local_tokenizer_files\target.spm",
                    @"D:\tools\opus-mt-onnx\my_local_tokenizer_files\vocab.json",
                    "<unk>",
                    "</s>",
                    "<pad>",
                    50
                );

                // C# equivalent of a List of BiasDefinitions:
                var SEQUENCE_BIAS_CONFIG = new List<BiasDefinition>
                {
                    new BiasDefinition
                    {
                        // The sequence is just the pad token ID
                        SequenceIds = new List<int> { tokenizer.PadTokenId }, 
                        // C# equivalent of -np.inf
                        BiasValue = float.NegativeInfinity
                    }
                };

                SequenceBiasLogitsProcessor biasProcessor = new SequenceBiasLogitsProcessor(SEQUENCE_BIAS_CONFIG);

                using var encoder_session = new InferenceSession(@"D:\tools\opus-mt-onnx\my_exported_onnx_model\encoder_model.onnx");
                using var decoder_session = new InferenceSession(@"D:\tools\opus-mt-onnx\my_exported_onnx_model\decoder_model.onnx");
                using var decoder_with_past_session = new InferenceSession(@"D:\tools\opus-mt-onnx\my_exported_onnx_model\decoder_with_past_model.onnx");

                var translated_text = RunMarianOnnxInference(
                    encoder_session,
                    decoder_session,
                    decoder_with_past_session,
                    tokenizer,
                    biasProcessor,
                    sentence,
                    50
                );

                Console.WriteLine(translated_text);
            }
        }

        public struct BertInput
        {
            public long[] InputIds { get; set; }
            public long[] AttentionMask { get; set; }
            public long[] TypeIds { get; set; }
        }

        internal class BertTokenizeProgram
        {
            static void BertMain(string[] args)
            {
                var sentence = "i am sad";
                Console.WriteLine(sentence);

                var vocabPath = @"D:\ai\onnx\emotion_analyzer-bert\vocab.txt";

                // Create Tokenizer and tokenize the sentence.
                var tokenizer = new BertUnasedCustomVocabulary(vocabPath);

                // Get the sentence tokens.
                var tokens = tokenizer.Tokenize(sentence);
                // Console.WriteLine(String.Join(", ", tokens));

                // Encode the sentence and pass in the count of the tokens in the sentence.
                var encoded = tokenizer.Encode(tokens.Count(), sentence);

                // Break out encoding to InputIds, AttentionMask and TypeIds from list of (input_id, attention_mask, type_id).
                var bertInput = new BertInput()
                {
                    InputIds = encoded.Select(t => t.InputIds).ToArray(),
                    AttentionMask = encoded.Select(t => t.AttentionMask).ToArray(),
                    TypeIds = encoded.Select(t => t.TokenTypeIds).ToArray(),
                };

                // Get path to model to create inference session.
                var modelPath = @"D:\ai\onnx\emotion_analyzer-bert\emotions-analyzer-bert.onnx";

                using var runOptions = new RunOptions();
                using var session = new InferenceSession(modelPath);

                // Create input tensors over the input data.
                using var inputIdsOrtValue = OrtValue.CreateTensorValueFromMemory(bertInput.InputIds,
                        new long[] { 1, bertInput.InputIds.Length });

                using var attMaskOrtValue = OrtValue.CreateTensorValueFromMemory(bertInput.AttentionMask,
                        new long[] { 1, bertInput.AttentionMask.Length });

                // Create input data for session. Request all outputs in this case.
                var inputs = new Dictionary<string, OrtValue>
            {
                { "input_ids", inputIdsOrtValue },
                { "attention_mask", attMaskOrtValue },
            };

                // Run session and send the input data in to get inference output. 
                using var output = session.Run(runOptions, inputs, session.OutputNames);
                // Get the Index of the Max value from the output lists.
                // We intentionally do not copy to an array or to a list to employ algorithms.
                // Hopefully, more algos will be available in the future for spans.
                // so we can directly read from native memory and do not duplicate data that
                // can be large for some models
                // Local function
                int GetMaxValueIndex(ReadOnlySpan<float> span)
                {
                    float maxVal = span[0];
                    int maxIndex = 0;
                    for (int i = 1; i < span.Length; ++i)
                    {
                        var v = span[i];
                        if (v > maxVal)
                        {
                            maxVal = v;
                            maxIndex = i;
                        }
                    }
                    return maxIndex;
                }

                var startLogits = output[0].GetTensorDataAsSpan<float>();
                int startIndex = GetMaxValueIndex(startLogits);

                var endLogits = output[output.Count - 1].GetTensorDataAsSpan<float>();
                int endIndex = GetMaxValueIndex(endLogits);

                var predictedTokens = tokens
                              .Skip(startIndex)
                              .Take(endIndex + 1 - startIndex)
                              .Select(o => tokenizer.IdToToken((int)o.VocabularyIndex))
                              .ToList();

                // Print the result.
                Console.WriteLine(String.Join(" ", predictedTokens));
            }
        }
    }