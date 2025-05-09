import type { IAgentRuntime } from '@elizaos/core';
import { z } from 'zod';
import { logger } from '@elizaos/core';
import type { MorpheusConfig } from './types';

/**
 * Schema for Morpheus environment configuration
 */
export const morpheusEnvSchema = z.object({
  MORPHEUS_API_KEY: z.string().min(1, 'Morpheus API key is required'),
  MORPHEUS_SMALL_MODEL: z.string().default('llama-3.2-3b'),
  MORPHEUS_LARGE_MODEL: z.string().default('llama-3.3-70b'),
  OPENAI_API_KEY: z.string().min(1, 'OpenAI API key is required for embeddings'),
  OPENAI_EMBEDDING_MODEL: z.string().default('text-embedding-3-small'),
  OPENAI_EMBEDDING_DIMENSIONS: z.number().int().positive().optional(),
});

/**
 * Helper function to get a setting from runtime or environment
 */
export function getSetting(runtime: IAgentRuntime, key: string, defaultValue?: string): string | undefined {
  return runtime.getSetting(key) ?? process.env[key] ?? defaultValue;
}

/**
 * Get Morpheus API base URL
 */
export function getBaseURL(): string {
  return 'http://api.mor.org/api/v1'; // Morpheus base URL (no trailing slash)
}

/**
 * Get Morpheus API key from runtime or environment
 */
export function getMorpheusApiKey(runtime: IAgentRuntime): string | undefined {
  return getSetting(runtime, 'MORPHEUS_API_KEY');
}

/**
 * Get small model name from runtime or environment
 */
export function getSmallModel(runtime: IAgentRuntime): string {
  return getSetting(runtime, 'MORPHEUS_SMALL_MODEL') ?? 'llama-3.2-3b';
}

/**
 * Get large model name from runtime or environment
 */
export function getLargeModel(runtime: IAgentRuntime): string {
  return getSetting(runtime, 'MORPHEUS_LARGE_MODEL') ?? 'llama-3.3-70b';
}

/**
 * Get OpenAI API key from runtime or environment
 */
export function getOpenAIApiKey(runtime: IAgentRuntime): string | undefined {
  return getSetting(runtime, 'OPENAI_API_KEY');
}

/**
 * Get OpenAI embedding model from runtime or environment
 */
export function getOpenAIEmbeddingModel(runtime: IAgentRuntime): string {
  return getSetting(runtime, 'OPENAI_EMBEDDING_MODEL') ?? 'text-embedding-3-small';
}

/**
 * Get OpenAI embedding dimensions from runtime or environment
 */
export function getOpenAIEmbeddingDimensions(runtime: IAgentRuntime): number | undefined {
  const dimsString = getSetting(runtime, 'OPENAI_EMBEDDING_DIMENSIONS');
  return dimsString ? parseInt(dimsString, 10) : undefined;
}

/**
 * OpenAI base URL for embeddings
 */
export const OPENAI_BASE_URL = 'https://api.openai.com/v1';

/**
 * Validates the Morpheus configuration against the schema
 */
export async function validateMorpheusConfig(runtime: IAgentRuntime): Promise<MorpheusConfig> {
  try {
    const config = {
      MORPHEUS_API_KEY: getMorpheusApiKey(runtime),
      MORPHEUS_SMALL_MODEL: getSmallModel(runtime),
      MORPHEUS_LARGE_MODEL: getLargeModel(runtime),
      OPENAI_API_KEY: getOpenAIApiKey(runtime),
      OPENAI_EMBEDDING_MODEL: getOpenAIEmbeddingModel(runtime),
      OPENAI_EMBEDDING_DIMENSIONS: getOpenAIEmbeddingDimensions(runtime),
    };

    return morpheusEnvSchema.parse(config);
  } catch (error) {
    if (error instanceof z.ZodError) {
      const errorMessages = error.errors
        .map((err) => `${err.path.join('.')}: ${err.message}`)
        .join('\n');
      throw new Error(`Morpheus configuration validation failed:\n${errorMessages}`);
    }
    throw error;
  }
} 