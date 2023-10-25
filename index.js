// Import dotenv for api_keys and fs for loading files
import { config } from 'dotenv'
import fs from 'fs'
config()

// Document Loaders
import { DirectoryLoader } from "langchain/document_loaders/fs/directory"
import { CSVLoader } from "langchain/document_loaders/fs/csv"
import { TextLoader } from 'langchain/document_loaders/fs/text'

// For splitting the text into chunks 
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter'

// LLM Chains and Model
import { ChatOpenAI } from "langchain/chat_models/openai"
import { ConversationChain } from "langchain/chains"

// Prompt Templates and Buffer Memory
import {ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder,
} from "langchain/prompts";

import { BufferMemory } from "langchain/memory";

// Vector database
import { HNSWLib } from "langchain/vectorstores/hnswlib"

// Embeddings for text
import { OpenAIEmbeddings } from "langchain/embeddings/openai"

const VECTOR_STORE_PATH = "Components_info.index"

// This variables should be taken from the api
const micro_controller = "ARDUINO"
const embedded_module = "PN532 NFC Module"
const question = `Tell me how to wire ${micro_controller} to ${embedded_module}`

const question2 = "What are the components that we can connect wit ARDUINO"


// Initialize document loaders
const loader = new DirectoryLoader("./documents", {
    ".csv": (path) => new CSVLoader(path),
    ".xlsx": (path) => new CSVLoader(path),
    ".txt": (path) => new TextLoader(path),
})

console.log("Loading documents...")
// Load the documents
const documents = await loader.load()
console.log("Loaded documents")

// Normalize the documents for better preprocessing
function normalizeDocuments(documents){
    return documents.map((document) => {
        if (typeof document.pageContent === 'string'){
            return document.pageContent
        }
        else if (Array.isArray(document.pageContent)){
            return document.pageContent.join('\n')
        }
    })
}


export const main_function = async () => {
    // Initialize the model
    const chat = new ChatOpenAI({ temperature: 0.1 })
    let vectorStore

    // Check if the vector store exists
    if (fs.existsSync(VECTOR_STORE_PATH)){
        console.log("Loading an existing vector store...")
        vectorStore = await HNSWLib.load(VECTOR_STORE_PATH, new OpenAIEmbeddings())
        console.log("Loaded an existing vector store")
    }
    else{
        console.log("Creating a new vector store...")

        // For splitting the text into chunks
        const textSplitter = new RecursiveCharacterTextSplitter({chunkSize: 1200, chunkOverlap: 200})
        const normalized_docs = normalizeDocuments(documents)
        const splitted_docs = await textSplitter.createDocuments(normalized_docs)

        // Create a new vector store
        vectorStore = await HNSWLib.fromDocuments(splitted_docs, new OpenAIEmbeddings())

        await vectorStore.save(VECTOR_STORE_PATH)
        console.log("New vector store created and saved")
    }

    // Promts for the chat
    const chatPrompt = ChatPromptTemplate.fromMessages([
    SystemMessagePromptTemplate.fromTemplate(
        "The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details how to wire different embedded components from its context. If the AI does not know the answer to a question, it truthfully says it does not know."
    ),
    new MessagesPlaceholder("history"),
    HumanMessagePromptTemplate.fromTemplate("{input}"),
    ]);



    // Initialize the chain
    console.log("Initializing the chain...")
    const chain = new ConversationChain({
        memory: new BufferMemory({ returnMessages: true, memoryKey: "history" }),
        prompt: chatPrompt,
        llm: chat,
    });

    // Ask the question
    console.log({question})
    const response = await chain.call({
        input: {
            question: question,
            vectorStore: vectorStore
        },
    });
    console.log({response})
}

// Run the main function
main_function()