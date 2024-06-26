import { NextRequest, NextResponse } from "next/server";
import { Message as VercelChatMessage, StreamingTextResponse } from "ai";

// import { ChatOpenAI } from "langchain/chat_models/openai";
import { ChatOllama } from "langchain/chat_models/ollama";
import { PromptTemplate } from "langchain/prompts";
import { HttpResponseOutputParser } from "langchain/output_parsers";

export const runtime = "edge";

const formatMessage = (message: VercelChatMessage) => {
  return `${message.role}: ${message.content}`;
};
let TEMPLATE: string;

if (process.env.LANGUAGE === 'en') {
  TEMPLATE = `You are an useful personal assistant named Jarvis. All responses must be extremely informative and useful for everyone.

  Current conversation:
  {chat_history}

  User: {input}
  AI:`;
} else if (process.env.LANGUAGE === 'th') {
  TEMPLATE = `คุณเป็นผู้ช่วยส่วนตัวที่มีประโยชน์ชื่อว่า Jarvis ทุกคำตอบต้องมีประโยชน์และเป็นประโยชน์อย่างมากสำหรับทุกคน

  บทสนทนาปัจจุบัน:
  {chat_history}

  ผู้ใช้: {input}
  AI:`;
}

/**
 * This handler initializes and calls a simple chain with a prompt,
 * chat model, and output parser. See the docs for more information:
 *
 * https://js.langchain.com/docs/guides/expression_language/cookbook#prompttemplate--llm--outputparser
 */
export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const messages = body.messages ?? [];
    const formattedPreviousMessages = messages.slice(0, -1).map(formatMessage);
    const currentMessageContent = messages[messages.length - 1].content;
    const prompt = PromptTemplate.fromTemplate(TEMPLATE);

    /**
     * You can also try e.g.:
     *
     * import { ChatAnthropic } from "langchain/chat_models/anthropic";
     * const model = new ChatAnthropic({});
     *
     * See a full list of supported models at:
     * https://js.langchain.com/docs/modules/model_io/models/
     */
    // const model = new ChatOpenAI({
    //   temperature: 0.8,
    // });
    const model = new ChatOllama({ 
      baseUrl: "http://localhost:11434",
      model: process.env.BASE_MODEL ?? "mistrallite", 
      temperature: 0.3
    });

    /**
     * Chat models stream message chunks rather than bytes, so this
     * output parser handles serialization and byte-encoding.
     */
    const outputParser = new HttpResponseOutputParser();

    /**
     * Can also initialize as:
     *
     * import { RunnableSequence } from "@langchain/core/runnables";
     * const chain = RunnableSequence.from([prompt, model, outputParser]);
     */
    const chain = prompt.pipe(model).pipe(outputParser);

    const stream = await chain.stream({
      chat_history: formattedPreviousMessages.join("\n"),
      input: currentMessageContent,
    });

    return new StreamingTextResponse(stream);
  } catch (e: any) {
    return NextResponse.json({ error: e.message }, { status: e.status ?? 500 });
  }
}
