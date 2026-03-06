import { createClient, type Client } from "@hey-api/client-fetch";
import type {
  CompletionRequest,
  CompletionResponse,
  ExtractionRequest,
  ExtractionResponse,
} from "./generated";

export type { CompletionRequest, CompletionResponse, ExtractionRequest, ExtractionResponse };

export interface LLMClientOptions {
  baseUrl: string;
  headers?: Record<string, string>;
}

export class LLMClient {
  private client: Client;

  constructor(options: LLMClientOptions) {
    this.client = createClient({
      baseUrl: options.baseUrl,
      headers: options.headers,
    });
  }

  async complete(request: CompletionRequest): Promise<CompletionResponse> {
    const response = await this.client.post<CompletionResponse>({
      url: "/v1/complete",
      body: request as unknown as Record<string, unknown>,
    });
    if (response.error) {
      throw new Error(`Complete request failed: ${JSON.stringify(response.error)}`);
    }
    return response.data as CompletionResponse;
  }

  async extract(request: ExtractionRequest): Promise<ExtractionResponse> {
    const response = await this.client.post<ExtractionResponse>({
      url: "/v1/extract",
      body: request as unknown as Record<string, unknown>,
    });
    if (response.error) {
      throw new Error(`Extract request failed: ${JSON.stringify(response.error)}`);
    }
    return response.data as ExtractionResponse;
  }

  async health(): Promise<{ status: string }> {
    const response = await this.client.get<{ status: string }>({
      url: "/health",
    });
    if (response.error) {
      throw new Error(`Health check failed: ${JSON.stringify(response.error)}`);
    }
    return response.data as { status: string };
  }
}
