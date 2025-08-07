# WebRAG

A web content retrieval and Q&A system that allows users to index web content and chat with an AI about it.

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application:**
   ```bash
   # Launch
   streamlit run app.py
   ```

3. **Configure in the app:**
   - Open your browser to `http://localhost:8501`
   - Add your OpenAI API key in the sidebar
   - Select your preferred models
   - Start indexing content!

## Features

- Web content extraction from URLs
- AI-powered Q&A with indexed content
- Content management interface
- Source citations with responses
- Progress tracking during processing
- Multiple OpenAI model support

## Requirements

- Python 3.8+
- OpenAI API key
- Internet connection for web scraping

## Configuration

Configure the application through the sidebar:

- **OpenAI API Key**: Your OpenAI API key
- **Embedding Model**: Choose from OpenAI's embedding models
- **Chat Model**: Select GPT model for responses
- **Temperature**: Control response creativity (0.0 = focused, 1.0 = creative)

## Usage

### Adding Content

1. Navigate to the "Add Content" tab
2. Paste URLs (one per line) in the text area
3. Configure crawling settings if needed
4. Click "Process URLs" to start indexing
5. Monitor progress and view results

### Chatting with Content

1. Switch to the "Chat" tab
2. Ask questions about your indexed content
3. View AI responses with source citations
4. Use follow-up questions for exploration
5. Export chat history when needed

### Managing Content

- View all indexed content in the Content Library
- Search and filter by title or domain
- Delete unwanted content
- Monitor storage usage in the sidebar

## Security & Privacy

- API keys are stored securely and not logged
- Web content is cached locally for performance
- No data is sent to third parties except OpenAI API
- Content deletion removes all associated data

## Troubleshooting

### Common Issues

1. **"API Key not configured"**
   - Add your OpenAI API key in the sidebar
   - Ensure the key is valid and has sufficient credits

2. **"No content extracted"**
   - Check if URLs are accessible
   - Verify the content is text-based (not PDFs, images, etc.)
   - Try different crawling settings

3. **"Vector store initialization failed"**
   - Ensure sufficient disk space
   - Check file permissions in the data directory
   - Try clearing browser cache and restarting
   - Retry if you just changed embedding model

4. **Slow processing**
   - Reduce max pages per domain
   - Lower crawl depth
   - Check internet connection speed

### Getting Help

If you encounter issues:
1. Check the console logs for error messages
2. Verify your OpenAI API key and credits
3. Ensure all dependencies are correctly installed
4. Try refreshing the browser page

## Performance Tips

- Start with a small number of URLs to test
- Use crawl depth of 1-2 for most use cases
- Monitor token usage in your OpenAI dashboard
- Clear old content periodically to save storage space

## License

MIT License. Please respect the terms of service of websites you scrape and the OpenAI API terms.

---
