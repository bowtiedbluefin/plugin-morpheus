import { createOpenAI } from '@ai-sdk/openai';
import type {
  Plugin,
  IAgentRuntime,
} from '@elizaos/core';
import { ModelType, logger } from '@elizaos/core';
import { generateObject, generateText } from 'ai';
import { FormData as NodeFormData, File as NodeFile } from 'formdata-node';
import {
  getBaseURL,
  getMorpheusApiKey,
  getSmallModel,
  getLargeModel,
  getOpenAIApiKey,
  getOpenAIEmbeddingModel,
  getOpenAIEmbeddingDimensions,
  OPENAI_BASE_URL,
  validateMorpheusConfig,
} from './environment';
import type { MorpheusApiRequestParams } from './types';

function createMorpheusClient(runtime: IAgentRuntime) {
  return createOpenAI({
    apiKey: getMorpheusApiKey(runtime),
    baseURL: getBaseURL(),
  });
}

const PLUGIN_VERSION = '1.1.2-obj-gen-fix'; // Updated version

async function generateMorpheusResponse(runtime: IAgentRuntime, params: any) {
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
      } as MorpheusApiRequestParams),
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
  models: {
    [ModelType.TEXT_LARGE]: async (runtime: IAgentRuntime, params: any) => {
      return generateMorpheusResponse(runtime, { ...params, modelType: ModelType.TEXT_LARGE });
    },
    [ModelType.TEXT_SMALL]: async (runtime: IAgentRuntime, params: any) => {
      return generateMorpheusResponse(runtime, { ...params, modelType: ModelType.TEXT_SMALL });
    },
    [ModelType.OBJECT_LARGE]: async (runtime: IAgentRuntime, params: any) => {
      const jsonPrompt = `${params.prompt}\n\nPlease provide your response strictly in JSON format. Do not include any explanatory text before or after the JSON object.`;
      const response = await generateMorpheusResponse(runtime, {
        runtime,
        prompt: jsonPrompt,
        modelType: ModelType.OBJECT_LARGE,
        temperature: params.temperature ?? 0,
        maxTokens: 8192,
        frequencyPenalty: 0.7,
        presencePenalty: 0.7,
        stopSequences: [],
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
    [ModelType.OBJECT_SMALL]: async (runtime: IAgentRuntime, params: any) => {
      const jsonPrompt = `${params.prompt}\n\nPlease provide your response strictly in JSON format. Do not include any explanatory text before or after the JSON object.`;
      const response = await generateMorpheusResponse(runtime, {
        runtime,
        prompt: jsonPrompt,
        modelType: ModelType.OBJECT_SMALL,
        temperature: params.temperature ?? 0,
        maxTokens: 8192,
        frequencyPenalty: 0.7,
        presencePenalty: 0.7,
        stopSequences: [],
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
    [ModelType.TEXT_EMBEDDING]: async (runtime: IAgentRuntime, params: any): Promise<number[]> => {
      logger.debug(`[plugin-morpheus/OpenAI Embed v${PLUGIN_VERSION}] Handler entered.`);
      const openaiApiKey = getOpenAIApiKey(runtime);
      const model = getOpenAIEmbeddingModel(runtime);
      const dimensions = getOpenAIEmbeddingDimensions(runtime);

      if (!openaiApiKey) {
        throw new Error('OPENAI_API_KEY is required for embeddings');
      }

      const openai = createOpenAI({
        apiKey: openaiApiKey,
        baseURL: OPENAI_BASE_URL,
      });

      try {
        const response = await fetch(`${OPENAI_BASE_URL}/embeddings`, {
          method: 'POST',
          headers: {
            Authorization: `Bearer ${openaiApiKey}`,
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            input: params.text || params.prompt || '',
            model: model,
            ...(dimensions && { dimensions }),
          }),
        });

        if (!response.ok) {
          throw new Error(`OpenAI API error: ${response.status} ${response.statusText}`);
        }

        const data = await response.json();
        return data.data[0].embedding;
      } catch (error) {
        logger.error('[plugin-morpheus] Error generating embedding:', error);
        throw error;
      }
    },
  },
  async init(config: Record<string, string>, runtime: IAgentRuntime) {
    logger.info(`[plugin-morpheus] Initializing v${PLUGIN_VERSION}`);
    try {
      // Validate configuration
      await validateMorpheusConfig(runtime);
    } catch (error: any) {
      logger.warn(`[plugin-morpheus] Configuration validation warning: ${error.message}`);
    }
    
    if (!getMorpheusApiKey(runtime)) {
      logger.warn(
        '[plugin-morpheus] MORPHEUS_API_KEY is not set - Morpheus text generation will fail'
      );
    }
    if (!getOpenAIApiKey(runtime)) {
      logger.warn('[plugin-morpheus] OPENAI_API_KEY is not set - Embeddings via OpenAI will fail');
    }
  },
};

export default morpheusPlugin;
