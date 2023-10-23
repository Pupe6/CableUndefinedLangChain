import { config } from 'dotenv'
config()

import { PromptTemplate } from 'langchain/prompts'
import { OpenAI } from "langchain/llms/openai"
import { LLMChain } from "langchain/chains"

const model = new OpenAI({temperature: 0.1})
const template = "Tell me how to wire {micro_controller} to {embedded_module}"

const prompt = new PromptTemplate({template, inputVariables: ['micro_controller', 'embedded_module']})

const chain = new LLMChain({llm: model, prompt})

const result = await chain.call({micro_controller: 'Arduino', embedded_module: 'PN532'})

console.log(result)
