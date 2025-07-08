# plugin-morpheus

Morpheus AI plugin for ElizaOS. This plugin provides integration with the Morpheus AI API for text generation capabilities, with a choice of OpenAI or Venice for embeddings.

## Installation & Setup

### From GitHub
```bash
# Install the plugin
npm install github:bowtiedbluefin/plugin-morpheus#dev

# Set required environment variables
export MORPHEUS_API_KEY="your-morpheus-api-key"
export EMBEDDING_PROVIDER="openai" # or "venice"
export EMBEDDING_API_KEY="your-embedding-api-key"
```

### Verify Installation
```bash
# Test plugin loading
node -e "import('@bowtiedbluefin/plugin-morpheus').then(m => console.log('Plugin loaded:', m.default.name))"
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
- `EMBEDDING_PROVIDER`: The embedding service to use. Must be either `openai` or `venice`.
- `EMBEDDING_API_KEY`: API key for your chosen embedding provider.
- `EMBEDDING_MODEL`: Optional. The embedding model to use. If not set, a default will be used (`text-embedding-3-small` for OpenAI, `text-embedding-bge-m3` for Venice).
- `EMBEDDING_DIMENSIONS`: Optional. Custom dimensions for embeddings (defaults to `1024`).

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
