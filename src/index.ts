import { createOpenAI } from '@ai-sdk/openai';
import type {
  ModelTypeName,
  ObjectGenerationParams,
  Plugin,
  TextEmbeddingParams,
} from '@elizaos/core';
import { type GenerateTextParams, ModelType, logger } from '@elizaos/core';
import { generateObject, generateText } from 'ai';
import { FormData as NodeFormData, File as NodeFile } from 'formdata-node';

// Morpheus-specific helper functions
function getSetting(runtime: any, key: string, defaultValue?: string): string | undefined {
  return runtime.getSetting(key) ?? process.env[key] ?? defaultValue;
}

function getBaseURL(): string {
  return 'http://api.mor.org/api/v1'; // Morpheus base URL (no trailing slash)
}

function getMorpheusApiKey(runtime: any): string | undefined {
  return getSetting(runtime, 'MORPHEUS_API_KEY');
}

function getSmallModel(runtime: any): string {
  return getSetting(runtime, 'MORPHEUS_SMALL_MODEL') ?? 'llama-3.2-3b';
}

function getLargeModel(runtime: any): string {
  return getSetting(runtime, 'MORPHEUS_LARGE_MODEL') ?? 'llama-3.3-70b';
}

// OpenAI-specific helper functions for embeddings
function getOpenAIApiKey(runtime: any): string | undefined {
  return getSetting(runtime, 'OPENAI_API_KEY');
}

function getOpenAIEmbeddingModel(runtime: any): string {
  return getSetting(runtime, 'OPENAI_EMBEDDING_MODEL') ?? 'text-embedding-3-small';
}

function getOpenAIEmbeddingDimensions(runtime: any): number | undefined {
  const dimsString = getSetting(runtime, 'OPENAI_EMBEDDING_DIMENSIONS');
  return dimsString ? parseInt(dimsString, 10) : undefined;
}

const OPENAI_BASE_URL = 'https://api.openai.com/v1';

function createMorpheusClient(runtime: any) {
  return createOpenAI({
    apiKey: getMorpheusApiKey(runtime),
    baseURL: getBaseURL(),
  });
}

const PLUGIN_VERSION = '1.1.2-obj-gen-fix'; // Updated version

async function generateMorpheusResponse(runtime: any, params: GenerateTextParams) {
  const morpheus = createMorpheusClient(runtime);
  const model =
    params.modelType === ModelType.TEXT_LARGE ? getLargeModel(runtime) : getSmallModel(runtime);

  try {
    const response = await fetch(`${getBaseURL()}/chat/completions`, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${getMorpheusApiKey(runtime)}`,
        'Content-Type': 'application/json',
        Accept: 'application/json',
      },
      body: JSON.stringify({
        model: model,
        messages: [
          {
            role: 'system',
            content: runtime.character.system ?? 'You are a helpful assistant.',
          },
          {
            role: 'user',
            content: params.prompt,
          },
        ],
        stream: true,
        temperature: params.temperature ?? 0.7,
        max_tokens: params.maxTokens ?? 8192,
        frequency_penalty: params.frequencyPenalty ?? 0.7,
        presence_penalty: params.presencePenalty ?? 0.7,
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error('No response body');
    }

    let fullText = '';
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value);
      const lines = chunk.split('\n');

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            const jsonStr = line.slice(6); // Remove 'data: ' prefix
            if (jsonStr.trim() === '') continue;

            const json = JSON.parse(jsonStr);
            if (json.choices?.[0]?.delta?.content) {
              fullText += json.choices[0].delta.content;
            }
          } catch (e) {
            logger.debug('[plugin-morpheus] Skipping non-JSON line:', line);
          }
        }
      }
    }

    return fullText.trim();
  } catch (error) {
    logger.error('[plugin-morpheus] Error generating response:', error);
    throw error;
  }
}

export const morpheusPlugin: Plugin = {
  name: 'morpheus',
  description: `Morpheus AI plugin (Handles Inference; Embeddings via OpenAI - v${PLUGIN_VERSION})`,
  config: {
    MORPHEUS_API_KEY: process.env.MORPHEUS_API_KEY,
    MORPHEUS_SMALL_MODEL: process.env.MORPHEUS_SMALL_MODEL,
    MORPHEUS_LARGE_MODEL: process.env.MORPHEUS_LARGE_MODEL,
    OPENAI_API_KEY: process.env.OPENAI_API_KEY,
    OPENAI_EMBEDDING_MODEL: process.env.OPENAI_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_DIMENSIONS: process.env.OPENAI_EMBEDDING_DIMENSIONS,
  },
  async init(_config, runtime) {
    logger.info(`[plugin-morpheus] Initializing v${PLUGIN_VERSION}`);
    if (!getMorpheusApiKey(runtime)) {
      logger.warn(
        '[plugin-morpheus] MORPHEUS_API_KEY is not set - Morpheus text generation will fail'
      );
    }
    if (!getOpenAIApiKey(runtime)) {
      logger.warn('[plugin-morpheus] OPENAI_API_KEY is not set - Embeddings via OpenAI will fail');
    }
  },
  models: {
    [ModelType.TEXT_LARGE]: async (runtime, params: GenerateTextParams) => {
      return generateMorpheusResponse(runtime, { ...params, modelType: ModelType.TEXT_LARGE });
    },
    [ModelType.TEXT_SMALL]: async (runtime, params: GenerateTextParams) => {
      return generateMorpheusResponse(runtime, { ...params, modelType: ModelType.TEXT_SMALL });
    },
    [ModelType.OBJECT_LARGE]: async (runtime, params: ObjectGenerationParams) => {
      const jsonPrompt = `${params.prompt}\n\nPlease provide your response strictly in JSON format. Do not include any explanatory text before or after the JSON object.`;
      const response = await generateMorpheusResponse(runtime, {
        prompt: jsonPrompt,
        modelType: ModelType.OBJECT_LARGE,
        temperature: params.temperature ?? 0,
        maxTokens: 8192,
        frequencyPenalty: 0.7,
        presencePenalty: 0.7,
        stopSequences: [],
        runtime,
      });

      try {
        // Clean the response to ensure we only parse the JSON part
        const jsonMatch = response.match(/\{[\s\S]*\}/);
        if (!jsonMatch) {
          throw new Error('No JSON object found in response');
        }
        return JSON.parse(jsonMatch[0]);
      } catch (error) {
        logger.error('[plugin-morpheus] Failed to parse JSON response:', { response, error });
        throw new Error('Model did not return valid JSON.');
      }
    },
    [ModelType.OBJECT_SMALL]: async (runtime, params: ObjectGenerationParams) => {
      const jsonPrompt = `${params.prompt}\n\nPlease provide your response strictly in JSON format. Do not include any explanatory text before or after the JSON object.`;
      const response = await generateMorpheusResponse(runtime, {
        prompt: jsonPrompt,
        modelType: ModelType.OBJECT_SMALL,
        temperature: params.temperature ?? 0,
        maxTokens: 8192,
        frequencyPenalty: 0.7,
        presencePenalty: 0.7,
        stopSequences: [],
        runtime,
      });

      try {
        // Clean the response to ensure we only parse the JSON part
        const jsonMatch = response.match(/\{[\s\S]*\}/);
        if (!jsonMatch) {
          throw new Error('No JSON object found in response');
        }
        return JSON.parse(jsonMatch[0]);
      } catch (error) {
        logger.error('[plugin-morpheus] Failed to parse JSON response:', { response, error });
        throw new Error('Model did not return valid JSON.');
      }
    },
    [ModelType.TEXT_EMBEDDING]: async (runtime, params: TextEmbeddingParams): Promise<number[]> => {
      logger.debug(`[plugin-morpheus/OpenAI Embed v${PLUGIN_VERSION}] Handler entered.`);
      const openaiApiKey = getOpenAIApiKey(runtime);
      const model = getOpenAIEmbeddingModel(runtime);
      const dimensions = getOpenAIEmbeddingDimensions(runtime);
      const hardcodedDimensionFallback = 1536;

      if (!params?.text || params.text.trim() === '') {
        logger.debug(
          `[plugin-morpheus/OpenAI Embed v${PLUGIN_VERSION}] Creating test embedding for initialization/empty text`
        );
        const initDimensions = dimensions ?? hardcodedDimensionFallback;
        const testVector = new Array(initDimensions).fill(0);
        testVector[0] = 0.1;
        return testVector;
      }

      if (!openaiApiKey) {
        logger.error(
          `[plugin-morpheus/OpenAI Embed v${PLUGIN_VERSION}] OPENAI_API_KEY is missing. Cannot generate embedding.`
        );
        const errorDims = dimensions ?? hardcodedDimensionFallback;
        const errorVector = new Array(errorDims).fill(0);
        errorVector[0] = 0.3;
        return errorVector;
      }

      logger.debug(
        `[plugin-morpheus/OpenAI Embed v${PLUGIN_VERSION}] Attempting OpenAI API call...`
      );
      try {
        const payload: { model: string; input: string; dimensions?: number } = {
          model: model,
          input: params.text,
        };
        if (dimensions !== undefined) {
          payload.dimensions = dimensions;
        }
        logger.debug(
          `[plugin-morpheus/OpenAI Embed v${PLUGIN_VERSION}] Calling ${OPENAI_BASE_URL}/embeddings with model ${model}`
        );

        const response = await fetch(`${OPENAI_BASE_URL}/embeddings`, {
          method: 'POST',
          headers: {
            Authorization: `Bearer ${openaiApiKey}`,
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(payload),
        });

        logger.debug(
          `[plugin-morpheus/OpenAI Embed v${PLUGIN_VERSION}] Received response status: ${response.status}`
        );
        if (!response.ok) {
          let errorBody = 'Could not parse error body';
          try {
            errorBody = await response.text();
          } catch (e) {
            /* ignore */
          }
          logger.error(
            `[plugin-morpheus/OpenAI Embed v${PLUGIN_VERSION}] OpenAI API error: ${response.status} - ${response.statusText}`,
            { errorBody }
          );
          throw new Error(`OpenAI API error: ${response.status} - ${response.statusText}`);
        }

        const data = (await response.json()) as {
          data: Array<{ embedding: number[] }>;
          usage: object;
          model: string;
        };
        logger.debug(
          `[plugin-morpheus/OpenAI Embed v${PLUGIN_VERSION}] Successfully parsed OpenAI response.`
        );

        if (!data?.data?.[0]?.embedding) {
          logger.error(
            `[plugin-morpheus/OpenAI Embed v${PLUGIN_VERSION}] No embedding returned from OpenAI API`,
            { responseData: data }
          );
          throw new Error('No embedding returned from OpenAI API');
        }

        const embedding = data.data[0].embedding;
        const embeddingDimensions = embedding.length;

        if (dimensions !== undefined && embeddingDimensions !== dimensions) {
          logger.warn(
            `[plugin-morpheus/OpenAI Embed v${PLUGIN_VERSION}] OpenAI Embedding dimensions mismatch: requested ${dimensions}, got ${embeddingDimensions}`
          );
        }

        logger.debug(
          `[plugin-morpheus/OpenAI Embed v${PLUGIN_VERSION}] Returning embedding with dimensions ${embeddingDimensions}.`
        );
        return embedding;
      } catch (error) {
        logger.error(
          `[plugin-morpheus/OpenAI Embed v${PLUGIN_VERSION}] Error during OpenAI embedding generation process:`,
          error
        );
        const errorDims = dimensions ?? hardcodedDimensionFallback;
        const errorVector = new Array(errorDims).fill(0);
        errorVector[0] = 0.2;
        return errorVector;
      }
    },
  },
};

export default morpheusPlugin;
