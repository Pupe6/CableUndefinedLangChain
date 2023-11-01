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
import { OpenAI } from "langchain/llms/openai"
import { RetrievalQAChain, loadQAStuffChain } from "langchain/chains";


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
    const model = new OpenAI({ temperature: 0.7, modelName: "gpt-3.5-turbo" })
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
        const textSplitter = new RecursiveCharacterTextSplitter({chunkSize: 1000, chunkOverlap: 100})
        const normalized_docs = normalizeDocuments(documents)
        const splitted_docs = await textSplitter.createDocuments(normalized_docs)

        // Create a new vector store
        vectorStore = await HNSWLib.fromDocuments(splitted_docs, new OpenAIEmbeddings())
        await vectorStore.save(VECTOR_STORE_PATH)
        console.log("New vector store created and saved")
    }


    const chain = new RetrievalQAChain({
    combineDocumentsChain: loadQAStuffChain(model),
    retriever: vectorStore.asRetriever(),
    returnSourceDocuments: true,
    });

    const res = await chain.call({
    query: "What do you know?",
    });

    console.log(res.text)
}

// Run the main function
main_function()