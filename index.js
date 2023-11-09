// Import dotenv for api_keys and fs for loading files
import { config } from "dotenv";
import fs from "fs";
import pkg from "xlsx";
const { readFile, utils, writeFile } = pkg;

config();

// Document Loaders
import { DirectoryLoader } from "langchain/document_loaders/fs/directory";
import { CSVLoader } from "langchain/document_loaders/fs/csv";

// For splitting the text into chunks
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

// LLM Chains and Model
import { OpenAI } from "langchain/llms/openai";
import { VectorDBQAChain } from "langchain/chains";

// Vector database
import { HNSWLib } from "langchain/vectorstores/hnswlib";
// Embeddings for text
import { OpenAIEmbeddings } from "langchain/embeddings/openai";

const VECTOR_STORE_PATH = "Components_info.index";

function convert_xlsx_to_csv(filePath) {
	const workbook = readFile(filePath);

	let worksheet = workbook.Sheets["Sheet1"];
	let jsonData = utils.sheet_to_json(worksheet, { raw: false, defval: null });
	let fileName = filePath.substring(0, filePath.length - 5);
	let new_worksheet = utils.json_to_sheet(jsonData);
	let new_workbook = utils.book_new();
	utils.book_append_sheet(new_workbook, new_worksheet, "csv_sheet");

	writeFile(new_workbook, fileName + ".csv");
	console.log("File converted to csv successfully!");
}

// Initialize document loaders
async function loadDocuments() {
	const loader = new DirectoryLoader("./documents", {
		".csv": path => new CSVLoader(path),
		".txt": path => new TextLoader(path),
	});
	console.log("Loading documents...");
	// Load the documents
	const documents = await loader.load();
	console.log("Loaded documents");
	return documents;
}

// Normalize the documents for better preprocessing
function normalizeDocuments(documents) {
	return documents.map(document => {
		if (typeof document.pageContent === "string") {
			return document.pageContent;
		} else if (Array.isArray(document.pageContent)) {
			return document.pageContent.join("\n");
		}
	});
}

export const main_function = async (micro_controller, embedded_module) => {
	let question = `Tell me how to wire ${micro_controller} to ${embedded_module}`;

	// Initialize the model
	const model = new OpenAI({ temperature: 0.05, modelName: "gpt-3.5-turbo" });
	let vectorStore;
	let splitted_docs;

	// Check if the vector store exists
	if (fs.existsSync(VECTOR_STORE_PATH)) {
		console.log("Loading an existing vector store...");
		vectorStore = await HNSWLib.load(
			VECTOR_STORE_PATH,
			new OpenAIEmbeddings()
		);
		console.log("Loaded an existing vector store");
	} else {
		console.log("Creating a new vector store...");
		convert_xlsx_to_csv("./documents/components_info.xlsx");
		// For splitting the text into chunks
		const textSplitter = new RecursiveCharacterTextSplitter({
			chunkSize: 1000,
			chunkOverlap: 100,
		});
		const documents = await loadDocuments();
		const normalized_docs = normalizeDocuments(documents);
		const splitted_docs = await textSplitter.createDocuments(
			normalized_docs
		);

		// Create a new vector store
		vectorStore = await HNSWLib.fromDocuments(
			splitted_docs,
			new OpenAIEmbeddings()
		);
		await vectorStore.save(VECTOR_STORE_PATH);
		console.log("New vector store created and saved");
	}

	const chain = VectorDBQAChain.fromLLM(model, vectorStore);

	const res = await chain.call({
		input_documents: splitted_docs,
		query: question,
	});

	console.log(res.text);
};
