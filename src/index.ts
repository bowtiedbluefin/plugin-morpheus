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

function getOpenAIApiKey(runtime: IAgentRuntime): string | undefined {
  return getSetting(runtime, 'OPENAI_API_KEY');
}

function getVeniceApiKey(runtime: IAgentRuntime): string | undefined {
  return getSetting(runtime, 'VENICE_API_KEY');
}

function getEmbeddingModel(runtime: IAgentRuntime): string {
  const provider = getEmbeddingProvider(runtime);
  if (provider === 'venice') {
    return (
      getSetting(runtime, 'VENICE_EMBEDDING_MODEL') ??
      'text-embedding-bge-m3'
    );
  }
  return (
    getSetting(runtime, 'OPENAI_EMBEDDING_MODEL') ?? 'text-embedding-3-small'
  );
}

function getEmbeddingDimensions(runtime: IAgentRuntime): number {
  const provider = getEmbeddingProvider(runtime);
  const key =
    provider === 'venice'
      ? 'VENICE_EMBEDDING_DIMENSIONS'
      : 'OPENAI_EMBEDDING_DIMENSIONS';
  const defaultValue = provider === 'venice' ? 1024 : 1536;
  const setting = getSetting(runtime, key);
  return setting ? parseInt(setting, 10) : defaultValue;
}

function getBaseURL(runtime: IAgentRuntime, provider: 'morpheus' | 'openai' | 'venice'): string {
  switch (provider) {
    case 'morpheus':
      return 'https://api.mor.org/v1';
    case 'openai':
      return getSetting(runtime, 'OPENAI_BASE_URL', 'https://api.openai.com/v1') as string;
    case 'venice':
       return getSetting(runtime, 'VENICE_BASE_URL', 'https://api.venice.ai/v1') as string;
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
  const apiKey =
    provider === 'venice'
      ? getVeniceApiKey(runtime)
      : getOpenAIApiKey(runtime);
  const baseURL =
    provider === 'venice'
      ? getBaseURL(runtime, 'venice')
      : getBaseURL(runtime, 'openai');

  if (!apiKey) {
    throw new Error(`API key for ${provider} is not configured.`);
  }

  return createOpenAI({ apiKey, baseURL });
}
// #endregion

// #region Object Generation
function getJsonRepairFunction(): (params: {
  text: string;
  error: unknown;
}) => Promise<string | null> {
  return async ({ text, error }) => {
    logger.warn('JSON parsing failed, attempting to repair...', {
      text,
      error,
    });
    // Placeholder for a more robust JSON repair mechanism
    const repaired = text.replace(/`/g, '').trim();
    return repaired;
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
      ...(params.schema && { schema: params.schema }),
      experimental_repairText: getJsonRepairFunction(),
    });
    return object as JSONValue;
  } catch (error) {
    if (error instanceof JSONParseError) {
      logger.error('Failed to parse JSON response from model', {
        error: error.message,
        text: error.text,
      });
      throw new Error(
        `Failed to generate valid JSON for ${modelType} using ${modelName}.`,
      );
    }
    throw error;
  }
}
// #endregion

// #region Plugin Definition
export const morpheusPlugin: Plugin = {
  name: 'morpheus',
  description:
    'Morpheus plugin for Text Generation, with optional Venice/OpenAI embeddings.',

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
