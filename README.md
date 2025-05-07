# plugin-morpheus

Morpheus AI plugin for ElizaOS. This plugin provides integration with the Morpheus AI API for text generation capabilities, with OpenAI for embeddings.

## Configuration

The plugin requires the following environment variables to be set:

- `MORPHEUS_API_KEY`: Your Morpheus AI API key
- `MORPHEUS_SMALL_MODEL`: The model to use for small text generation (defaults to 'default')
- `MORPHEUS_LARGE_MODEL`: The model to use for large text generation (defaults to 'default')
- `OPENAI_API_KEY`: Your OpenAI API key (required for embeddings)
- `OPENAI_EMBEDDING_MODEL`: The OpenAI model to use for embeddings (defaults to 'text-embedding-3-small')
- `OPENAI_EMBEDDING_DIMENSIONS`: Optional custom dimensions for embeddings

## Features

- Text Generation using Morpheus Compute Marketplace
- Configurable Model Selection from list at http://api.mor.org/api/v1/models/
- Text embeddings via OpenAI

## Usage

To use the Morpheus plugin, add it to your ElizaOS configuration:

```typescript
import { morpheusPlugin } from '@elizaos/plugin-morpheus';

// Add to your plugins array
const plugins = [
  morpheusPlugin,
  // ... other plugins
];
```

## Building

To build the plugin:

```bash
bun run build
```

To watch for changes during development:

```bash
bun run dev
```

## License

MIT
