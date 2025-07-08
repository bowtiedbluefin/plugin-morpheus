# plugin-morpheus

Morpheus AI plugin for ElizaOS. This plugin provides integration with the Morpheus AI API for text generation capabilities, with a choice of OpenAI or Venice for embeddings.

## Installation & Setup

### From GitHub
```bash
# Install the plugin
npm install github:bowtiedbluefin/plugin-morpheus

# Set required environment variables
export MORPHEUS_API_KEY="your-morpheus-api-key"

# For OpenAI Embeddings (default)
export EMBEDDING_PROVIDER="openai"
export OPENAI_API_KEY="your-openai-api-key"

# For Venice Embeddings
export EMBEDDING_PROVIDER="venice"
export VENICE_API_KEY="your-venice-api-key"
```

### Verify Installation
```bash
# Test plugin loading
node -e "import('@elizaos/plugin-morpheus').then(m => console.log('Plugin loaded:', m.default.name))"
```

### Integration with Eliza
Add to your character's `plugins` array:
```json
{
  "plugins": [
    "@elizaos/plugin-morpheus"
  ]
}
```

Or ensure `MORPHEUS_API_KEY` is set for automatic loading in the default Eliza character.

## Configuration

The plugin can be configured via environment variables or runtime settings:

- `MORPHEUS_API_KEY`: Your Morpheus AI API key.
- `MORPHEUS_SMALL_MODEL`: The model for small text generation (defaults to `mistral-31-24b`).
- `MORPHEUS_LARGE_MODEL`: The model for large text generation (defaults to `qwen3-235b`).
- `EMBEDDING_PROVIDER`: The embedding service to use, either `openai` (default) or `venice`.

### OpenAI Embeddings
- `OPENAI_API_KEY`: Your OpenAI API key.
- `OPENAI_EMBEDDING_MODEL`: The OpenAI model for embeddings (defaults to `text-embedding-3-small`).
- `OPENAI_EMBEDDING_DIMENSIONS`: Optional dimensions for OpenAI embeddings.

### Venice Embeddings
- `VENICE_API_KEY`: Your Venice API key.
- `VENICE_EMBEDDING_MODEL`: The Venice model for embeddings (defaults to `text-embedding-bge-m3`).
- `VENICE_EMBEDDING_DIMENSIONS`: Optional dimensions for Venice embeddings.

## Features

- Text Generation using Morpheus Compute Marketplace.
- Configurable Model Selection from the list at http://api.mor.org/api/v1/models/.
- Choice of Text Embeddings via OpenAI or Venice.
- Robust Object Generation with JSON repair.

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

## Development

To work on the plugin:

```bash
npm run dev
```

To format code:

```bash
npm run format
```

## License

MIT
