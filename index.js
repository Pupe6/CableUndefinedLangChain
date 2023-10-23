// Import dotenv for api_keys and fs for loading files
import { config } from 'dotenv'
import fs from 'fs'
config()

// Prompt Template for custom prompt questions
import { PromptTemplate } from 'langchain/prompts'

// Document Loaders
import { DirectoryLoader } from "langchain/document_loaders/fs/directory"
import { CSVLoader } from "langchain/document_loaders/fs/csv"
import { TextLoader } from 'langchain/document_loaders/fs/text'

// For splitting the text into chunks 
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter'

// LLM Chains and Model
import { OpenAI } from "langchain/llms/openai"
import { RetrievalQAChain } from "langchain/chains"

// Vector database
import { HNSWLib } from "langchain/vectorstores/hnswlib"

// Embeddings for text
import { OpenAIEmbeddings } from "langchain/embeddings/openai"

// Initialize document loaders
const loader = new DirectoryLoader("./documents", {
    ".csv": (path) => new CSVLoader(path),
    ".xlsx": (path) => new CSVLoader(path),
    ".txt": (path) => new TextLoader(path),
})


const template = "Tell me how to wire {micro_controller} to {embedded_module}"

const prompt = new PromptTemplate({template, inputVariables: ['micro_controller', 'embedded_module']})

