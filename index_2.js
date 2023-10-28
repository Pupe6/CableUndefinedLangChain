import * as fs from "fs";
import { PromptTemplate } from "langchain/prompts";
import { RunnableSequence } from "langchain/schema/runnable";
import { StringOutputParser } from "langchain/schema/output_parser";


import { config } from 'dotenv'
config()

// Document Loaders
import { DirectoryLoader } from "langchain/document_loaders/fs/directory"
import { CSVLoader } from "langchain/document_loaders/fs/csv"
import { TextLoader } from 'langchain/document_loaders/fs/text'
// For splitting the text into chunks 
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter'

// LLM Chains and Model
import { ChatOpenAI } from "langchain/chat_models/openai"

// Vector database
import { HNSWLib } from "langchain/vectorstores/hnswlib"
// Embeddings for text
import { OpenAIEmbeddings } from "langchain/embeddings/openai"

const VECTOR_STORE_PATH = "Components_info.index"
// This variables should be taken from the api
const micro_controller = "Rapberry Pi Pico"
const embedded_module = "Buzzer"
const questionOne = `Tell me how to wire ${micro_controller} to ${embedded_module}`

const questionTwo = "What are the others components that we can connect with it?"

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
    const model = new ChatOpenAI({ temperature: 0.1, modelName: "gpt-3.5-turbo" })
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

    // init retriever
    const retriever = vectorStore.asRetriever()
    
    const formatChatHistory = ( human, ai, previousChatHistory ) => {
        const newInteraction = `Human: ${human}\nAI: ${ai}`;
        if (!previousChatHistory) {
            return newInteraction;
        }
        return `${previousChatHistory}\n\n${newInteraction}`;
    };


    const questionPrompt = PromptTemplate.fromTemplate(
        `Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
        ----------------
        CHAT HISTORY: {chatHistory}
        ----------------
        QUESTION: {question}
        ----------------
        Helpful Answer:`
    );

    const chain = RunnableSequence.from([
    {
        question: (input) =>
            input.question,
        chatHistory: (input) =>
            input.chatHistory ?? "",
    },
    questionPrompt,
    model,
    new StringOutputParser(),
    ]);

    // Ask the question
    const resultOne = await chain.invoke({
        question: questionOne,
    });

    console.log({ resultOne });
    /**
     * {
     *   resultOne: 'The president thanked Justice Breyer for his service and described him as an Army veteran, Constitutional scholar, and retiring Justice of the United States Supreme Court.'
     * }
     */

    const resultTwo = await chain.invoke({
        chatHistory: formatChatHistory(resultOne, questionOne),
        question: questionTwo,
    });

    console.log({ resultTwo });

}

// Call the main function
main_function()
