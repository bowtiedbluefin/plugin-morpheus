/**Configuration for the Morpheus plugin*/
export interface MorpheusConfig {
  MORPHEUS_API_KEY: string;
  MORPHEUS_SMALL_MODEL: string;
  MORPHEUS_LARGE_MODEL: string;
  OPENAI_API_KEY: string;
  OPENAI_EMBEDDING_MODEL: string;
  OPENAI_EMBEDDING_DIMENSIONS?: number;
}

/**Parameters for the Morpheus API request*/
export interface MorpheusApiRequestParams {
  model: string;
  messages: Array<{
    role: string;
    content: string;
  }>;
  stream: boolean;
  temperature?: number;
  max_tokens?: number;
  frequency_penalty?: number;
  presence_penalty?: number;
} 