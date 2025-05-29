# plugin-morpheus

Morpheus AI plugin for ElizaOS. This plugin provides integration with the Morpheus AI API for text generation capabilities, with OpenAI for embeddings.

## Installation & Setup

### From GitHub
```bash
# Install the plugin
npm install github:bowtiedbluefin/plugin-morpheus

# Set required environment variables
export MORPHEUS_API_KEY="your-morpheus-api-key"
export MORPHEUS_SMALL_MODEL="llama-3.3-70b"
export MORPHEUS_LARGE_MODEL="llama-3.3-70b"
export OPENAI_API_KEY="your-openai-api-key"
export OPENAI_EMBEDDING_MODEL="text-embedding-3-small"
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

Or ensure `MORPHEUS_API_KEY` is set for automatic loading in default Eliza character.

## Configuration

The plugin requires the following environment variables to be set:

- `MORPHEUS_API_KEY`: Your Morpheus AI API key
- `MORPHEUS_SMALL_MODEL`: The model to use for small text generation (defaults to 'llama-3.3-70b')
- `MORPHEUS_LARGE_MODEL`: The model to use for large text generation (defaults to 'llama-3.3-70b')
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
