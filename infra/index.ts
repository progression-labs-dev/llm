import * as pulumi from "@pulumi/pulumi";
import * as gcp from "@pulumi/gcp";
import { defineConfig, createSecret, createContainer } from "@progression-labs/infra";

// Get configuration
const pulumiConfig = new pulumi.Config();
const environment = pulumiConfig.require("environment");
const gcpConfig = new pulumi.Config("gcp");
const project = gcpConfig.require("project");
const region = gcpConfig.require("region");

// Initialize infra package config
defineConfig({
  cloud: "gcp",
  region: region,
  project: "llm-gateway",
  environment: environment,
});

// Naming convention
const namePrefix = `llm-gateway-${environment}`;

// Common labels
const labels = {
  environment,
  service: "llm-gateway",
  "managed-by": "pulumi",
};

// =============================================================================
// Enable required APIs
// =============================================================================
const enabledApis = [
  "run.googleapis.com",
  "secretmanager.googleapis.com",
  "artifactregistry.googleapis.com",
  "iam.googleapis.com",
].map((api) => new gcp.projects.Service(`enable-${api.split('.')[0]}`, {
  project,
  service: api,
  disableOnDestroy: false,
}));

// =============================================================================
// Artifact Registry
// =============================================================================
const artifactRegistry = new gcp.artifactregistry.Repository(`${namePrefix}-repo`, {
  repositoryId: namePrefix.toLowerCase().replace(/[^a-z0-9-]/g, "-"),
  location: region,
  format: "DOCKER",
  description: "Docker repository for llm-gateway",
  labels,
}, { dependsOn: enabledApis });

// =============================================================================
// Secrets (using @progression-labs/infra)
// =============================================================================
const openaiKey = createSecret("openai-api-key");
const anthropicKey = createSecret("anthropic-api-key");
const langfusePublicKey = createSecret("langfuse-public-key");
const langfuseSecretKey = createSecret("langfuse-secret-key");

// =============================================================================
// Cloud Run Service (using @progression-labs/infra)
// =============================================================================
const llmGateway = createContainer("api", {
  image: "gcr.io/cloudrun/hello", // Placeholder — updated after first push
  port: 8000,
  size: "medium",
  minInstances: 0,
  maxInstances: 5,
  environment: {
    SERVICE_NAME: "llm-gateway",
    SERVICE_ENVIRONMENT: environment,
    DEFAULT_MODEL: "gpt-4o",
    DEFAULT_TIMEOUT: "60",
    DEFAULT_MAX_RETRIES: "3",
  },
  link: [openaiKey, anthropicKey, langfusePublicKey, langfuseSecretKey],
});

// =============================================================================
// Outputs
// =============================================================================
export const serviceUrl = llmGateway.url;
export const serviceName = llmGateway.serviceArn;
export const artifactRegistryUrl = pulumi.interpolate`${region}-docker.pkg.dev/${project}/${artifactRegistry.repositoryId}`;
