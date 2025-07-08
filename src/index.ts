import { createOpenAI } from '@ai-sdk/openai';
import type {
  IAgentRuntime,
  ObjectGenerationParams,
  Plugin,
  TextEmbeddingParams,
} from '@elizaos/core';
import { ModelType, type GenerateTextParams, logger } from '@elizaos/core';
import {
  JSONParseError,
  type JSONValue,
  embed,
  generateObject,
  generateText,
} from 'ai';

// #region Configuration
function getSetting(
  runtime: IAgentRuntime,
  key: string,
  defaultValue?: string,
): string | undefined {
  return runtime.getSetting(key) ?? process.env[key] ?? defaultValue;
}

function getMorpheusApiKey(runtime: IAgentRuntime): string | undefined {
  return getSetting(runtime, 'MORPHEUS_API_KEY');
}

function getSmallModel(runtime: IAgentRuntime): string {
  return getSetting(runtime, 'MORPHEUS_SMALL_MODEL', 'mistral-31-24b') as string;
}

function getLargeModel(runtime: IAgentRuntime): string {
  return getSetting(runtime, 'MORPHEUS_LARGE_MODEL', 'qwen3-235b') as string;
}

function getEmbeddingProvider(runtime: IAgentRuntime): 'openai' | 'venice' {
  return (
    (getSetting(runtime, 'EMBEDDING_PROVIDER') as 'openai' | 'venice') ??
    'openai'
  );
}

function getEmbeddingApiKey(runtime: IAgentRuntime): string | undefined {
  return getSetting(runtime, 'EMBEDDING_API_KEY');
}

function getEmbeddingModel(runtime: IAgentRuntime): string {
  const userModel = getSetting(runtime, 'EMBEDDING_MODEL');
  if (userModel) {
    return userModel;
  }
  const provider = getEmbeddingProvider(runtime);
  return provider === 'venice'
    ? 'text-embedding-bge-m3'
    : 'text-embedding-3-small';
}

function getEmbeddingDimensions(runtime: IAgentRuntime): number {
  const setting = getSetting(runtime, 'EMBEDDING_DIMENSIONS');
  return setting ? parseInt(setting, 10) : 1024;
}

function getBaseURL(runtime: IAgentRuntime, provider: 'morpheus' | 'openai' | 'venice'): string {
  switch (provider) {
    case 'morpheus':
      return 'https://api.mor.org/v1';
    case 'openai':
      return getSetting(runtime, 'OPENAI_BASE_URL', 'https://api.openai.com/v1') as string;
    case 'venice':
       return getSetting(runtime, 'VENICE_BASE_URL', 'https://api.venice.ai/api/v1') as string;
    default:
      return '';
  }
}
// #endregion

// #region API Clients
function createMorpheusClient(runtime: IAgentRuntime) {
  return createOpenAI({
    apiKey: getMorpheusApiKey(runtime),
    baseURL: getBaseURL(runtime, 'morpheus'),
  });
}

function createEmbeddingClient(runtime: IAgentRuntime) {
  const provider = getEmbeddingProvider(runtime);
  const apiKey = getEmbeddingApiKey(runtime);
  const baseURL =
    provider === 'venice'
      ? getBaseURL(runtime, 'venice')
      : getBaseURL(runtime, 'openai');

  if (!apiKey) {
    throw new Error(`EMBEDDING_API_KEY is not configured for provider ${provider}.`);
  }

  return createOpenAI({ apiKey, baseURL });
}
// #endregion

// #region Object Generation
function getJsonRepairFunction(): (params: {
  text: string;
  error: unknown;
}) => Promise<string | null> {
  return async ({ text, error }: { text: string; error: unknown }) => {
    try {
      if (error instanceof JSONParseError) {
        const cleanedText = text.replace(/```json\\n|\\n```|```/g, "");
        JSON.parse(cleanedText);
        return cleanedText;
      }
      return null;
    } catch (jsonError: unknown) {
      const message =
        jsonError instanceof Error ? jsonError.message : String(jsonError);
      logger.warn(`Failed to repair JSON text: ${message}`);
      return null;
    }
  };
}

async function generateObjectByModelType(
  runtime: IAgentRuntime,
  params: ObjectGenerationParams,
  modelType: string,
  getModelFn: (runtime: IAgentRuntime) => string,
): Promise<JSONValue> {
  const morpheus = createMorpheusClient(runtime);
  const modelName = getModelFn(runtime);
  logger.log(`[Morpheus] Using ${modelType} model: ${modelName}`);

  try {
    const { object } = await generateObject({
      model: morpheus.languageModel(modelName),
      output: 'no-schema',
      prompt: params.prompt,
      temperature: params.temperature ?? 0,
      experimental_repairText: getJsonRepairFunction(),
    });
    return object as JSONValue;
  } catch (error) {
    if (error instanceof JSONParseError) {
      logger.error(`[generateObject] Failed to parse JSON: ${error.message}`);

      const repairFunction = getJsonRepairFunction();
      const repairedJsonString = await repairFunction({
        text: error.text,
        error,
      });

      if (repairedJsonString) {
        try {
          const repairedObject = JSON.parse(repairedJsonString);
          logger.info("[generateObject] Successfully repaired JSON.");
          return repairedObject;
        } catch (repairParseError: unknown) {
          const message =
            repairParseError instanceof Error
              ? repairParseError.message
              : String(repairParseError);
          logger.error(
            `[generateObject] Failed to parse repaired JSON: ${message}`
          );
          throw repairParseError;
        }
      } else {
        logger.error("[generateObject] JSON repair failed.");
        throw error;
      }
    } else {
      const message = error instanceof Error ? error.message : String(error);
      logger.error(`[generateObject] Unknown error: ${message}`);
      throw error;
    }
  }
}
// #endregion

// #region Plugin Definition
export const morpheusPlugin: Plugin = {
  name: 'morpheus',
  description:
    'Morpheus plugin for Text Generation, with optional embedding support via Venice or OpenAI.',
  config: {
    MORPHEUS_API_KEY: process.env.MORPHEUS_API_KEY,
    MORPHEUS_SMALL_MODEL: process.env.MORPHEUS_SMALL_MODEL,
    MORPHEUS_LARGE_MODEL: process.env.MORPHEUS_LARGE_MODEL,
    EMBEDDING_PROVIDER: process.env.EMBEDDING_PROVIDER,
    EMBEDDING_API_KEY: process.env.EMBEDDING_API_KEY,
    EMBEDDING_MODEL: process.env.EMBEDDING_MODEL,
    EMBEDDING_DIMENSIONS: process.env.EMBEDDING_DIMENSIONS,
  },
  async init(_config, runtime) {
    logger.info('[plugin-morpheus] Initializing...');
    if (!getMorpheusApiKey(runtime)) {
      logger.warn(
        '[plugin-morpheus] MORPHEUS_API_KEY is not set - Morpheus functionality will fail',
      );
    }
  },

  models: {
    [ModelType.TEXT_LARGE]: async (
      runtime,
      {
        prompt,
        stopSequences = [],
        maxTokens = 8192,
        temperature = 0.7,
      }: GenerateTextParams,
    ) => {
      const morpheus = createMorpheusClient(runtime);
      const model = getLargeModel(runtime);
      const { text: morpheusResponse } = await generateText({
        model: morpheus.languageModel(model),
        prompt,
        system: runtime.character.system ?? undefined,
        temperature,
        maxTokens,
        stopSequences,
      });
      return morpheusResponse;
    },
    [ModelType.TEXT_SMALL]: async (
      runtime,
      {
        prompt,
        stopSequences = [],
        maxTokens = 4096,
        temperature = 0.7,
      }: GenerateTextParams,
    ) => {
      const morpheus = createMorpheusClient(runtime);
      const model = getSmallModel(runtime);
      const { text: morpheusResponse } = await generateText({
        model: morpheus.languageModel(model),
        prompt,
        system: runtime.character.system ?? undefined,
        temperature,
        maxTokens,
        stopSequences,
      });
      return morpheusResponse;
    },
    [ModelType.OBJECT_LARGE]: (runtime, params) =>
      generateObjectByModelType(
        runtime,
        params,
        ModelType.OBJECT_LARGE,
        getLargeModel,
      ),
    [ModelType.OBJECT_SMALL]: (runtime, params) =>
      generateObjectByModelType(
        runtime,
        params,
        ModelType.OBJECT_SMALL,
        getSmallModel,
      ),
    [ModelType.TEXT_EMBEDDING]: async (
      runtime: IAgentRuntime,
      params: TextEmbeddingParams,
    ): Promise<number[]> => {
      const provider = getEmbeddingProvider(runtime);
      logger.debug(`[Morpheus] Using ${provider} for embeddings.`);
      const client = createEmbeddingClient(runtime);
      const model = getEmbeddingModel(runtime);
      const dimensions = getEmbeddingDimensions(runtime);

      if (!params.text || params.text.trim() === '') {
        logger.debug(
          `[Morpheus] Creating test embedding for provider: ${provider}`,
        );
        return new Array(dimensions).fill(0.001);
      }

      try {
        const { embedding } = await embed({
          model: client.embedding(model, { dimensions }),
          value: params.text,
        });
        return embedding;
      } catch (error: any) {
        logger.error(`Error generating embedding with ${provider}:`, error);
        throw new Error(
          `Failed to generate embedding: ${error.message || 'Unknown error'}`,
        );
      }
    },
  },
};

export default morpheusPlugin;
