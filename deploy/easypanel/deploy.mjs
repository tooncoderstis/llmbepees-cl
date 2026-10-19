#!/usr/bin/env node
// Helper deploy EasyPanel via MCP (Streamable HTTP JSON-RPC) — tanpa dependensi.
// Env: EASYPANEL_API_KEY (wajib), EASYPANEL_BASE_URL (mis. https://panel.example.com).
// Lihat: templates/deploy/easypanel/RUNBOOK.md

import { readFileSync } from "node:fs";

const BASE = (process.env.EASYPANEL_BASE_URL || "").replace(/\/+$/, "");
const KEY = process.env.EASYPANEL_API_KEY || "";
const MCP_URL = BASE ? `${BASE}/api/mcp` : "";

const RESOURCES = {
  cpuLimit: 0,
  cpuReservation: 0,
  memoryLimit: 0,
  memoryReservation: 0,
};

let requestId = 1;
let sessionId = null;
let initialized = false;

const ALPHANUM =
  "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

function randomPassword(length = 32) {
  let out = "";

  for (let i = 0; i < length; i += 1) {
    out += ALPHANUM[Math.floor(Math.random() * ALPHANUM.length)];
  }

  return out;
}

function parseBody(text) {
  const trimmed = text.trim();

  if (trimmed.startsWith("{")) return JSON.parse(trimmed);

  for (const line of trimmed.split(/\r?\n/)) {
    if (!line.startsWith("data:")) continue;

    const payload = line.slice(5).trim();

    if (!payload || payload === "[DONE]") continue;

    const obj = JSON.parse(payload);

    if (obj.result !== undefined || obj.error !== undefined) return obj;
  }

  throw new Error(`Respons MCP tidak dapat di-parse: ${trimmed.slice(0, 200)}`);
}

async function rpc(method, params) {
  const headers = {
    "content-type": "application/json",
    accept: "application/json, text/event-stream",
    authorization: `Bearer ${KEY}`,
  };

  if (sessionId) headers["mcp-session-id"] = sessionId;

  const response = await fetch(MCP_URL, {
    method: "POST",
    headers,
    body: JSON.stringify({ jsonrpc: "2.0", id: requestId++, method, params }),
  });

  const session = response.headers.get("mcp-session-id");

  if (session) sessionId = session;

  if (!response.ok) {
    throw new Error(`HTTP ${response.status} dari MCP: ${await response.text()}`);
  }

  const json = parseBody(await response.text());

  if (json.error) throw new Error(json.error.message || "Error MCP");

  return json.result;
}

async function ensureInit() {
  if (initialized) return;

  if (!KEY) throw new Error("EASYPANEL_API_KEY belum diset.");
  if (!BASE) throw new Error("EASYPANEL_BASE_URL belum diset.");

  await rpc("initialize", {
    protocolVersion: "2024-11-05",
    capabilities: {},
    clientInfo: { name: "aasa-deploy", version: "1.0" },
  });

  await fetch(MCP_URL, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      accept: "application/json, text/event-stream",
      authorization: `Bearer ${KEY}`,
      ...(sessionId ? { "mcp-session-id": sessionId } : {}),
    },
    body: JSON.stringify({ jsonrpc: "2.0", method: "notifications/initialized" }),
  }).catch(() => undefined);

  initialized = true;
}

async function callTool(name, args) {
  await ensureInit();

  const result = await rpc("tools/call", { name, arguments: args });

  return result;
}

function textOf(result) {
  const content = result?.content ?? [];
  const first = content.find((item) => item.type === "text");

  if (!first) return result;

  try {
    return JSON.parse(first.text);
  } catch {
    return first.text;
  }
}

async function runProcedure(procedure, input, risk) {
  const tool =
    risk === "query"
      ? "execute_query"
      : risk === "destructive"
        ? "execute_destructive"
        : "execute_mutation";

  const result = await callTool(tool, { procedure, input });

  if (result?.isError) {
    throw new Error(`Gagal '${procedure}': ${result.content?.[0]?.text ?? ""}`);
  }

  return textOf(result);
}

async function search(query) {
  return textOf(await callTool("search_procedures", { query, limit: 5 }));
}

function parseArgs(argv) {
  const positional = [];
  const flags = {};

  for (let i = 0; i < argv.length; i += 1) {
    const token = argv[i];

    if (token.startsWith("--")) {
      const key = token.slice(2);
      const next = argv[i + 1];

      if (next === undefined || next.startsWith("--")) {
        flags[key] = true;
      } else {
        flags[key] = next;
        i += 1;
      }
    } else {
      positional.push(token);
    }
  }

  return { positional, flags };
}

function envFromFlags(flags) {
  if (flags["env-file"]) {
    return readFileSync(flags["env-file"], "utf8");
  }

  if (typeof flags.env === "string") return flags.env;

  return undefined;
}

const HELP = `Penggunaan: node deploy.mjs <perintah> [args]

  list
  search <query>
  create-project <name>
  create-postgres <project> <service> [--user U] [--db D] [--password P]
  create-redis <project> <service> [--password P]
  create-app <project> <service> --image IMG [--env-file F | --env "K=V"]
        [--registry-user U --registry-password P]
  create-domain <project> <service> <host> [--port 3000]
  set-env <project> <service> --env-file F | --env "K=V"
  deploy <project> <service> [--force]
  inspect <project> <service> [--type app|postgres|redis]
  raw <query|mutation|destructive> <procedure> <jsonInput>

Env: EASYPANEL_API_KEY (wajib), EASYPANEL_BASE_URL (mis. https://panel.example.com)`;

async function main() {
  const { positional, flags } = parseArgs(process.argv.slice(2));
  const command = positional[0];

  if (!command || flags.help || command === "help") {
    console.log(HELP);
    return;
  }

  let out;

  switch (command) {
    case "list":
      out = await runProcedure("listProjectsAndServices", {}, "query");
      break;

    case "search":
      out = await search(positional.slice(1).join(" "));
      break;

    case "create-project":
      out = await runProcedure(
        "createProject",
        { name: positional[1] },
        "mutation",
      );
      break;

    case "create-postgres": {
      const [, project, service] = positional;
      out = await runProcedure(
        "createPostgresService",
        {
          projectName: project,
          serviceName: service,
          user: flags.user ?? service,
          databaseName: flags.db ?? service,
          password: flags.password ?? randomPassword(),
          resources: RESOURCES,
        },
        "mutation",
      );
      break;
    }

    case "create-redis": {
      const [, project, service] = positional;
      out = await runProcedure(
        "createRedisService",
        {
          projectName: project,
          serviceName: service,
          password: flags.password ?? randomPassword(),
          resources: RESOURCES,
        },
        "mutation",
      );
      break;
    }

    case "create-app": {
      const [, project, service] = positional;
      const source = { type: "image", image: flags.image };

      if (flags["registry-user"]) {
        source.username = flags["registry-user"];
        source.password = flags["registry-password"];
      }

      out = await runProcedure(
        "createAppService",
        {
          projectName: project,
          serviceName: service,
          source,
          env: envFromFlags(flags),
          resources: RESOURCES,
        },
        "mutation",
      );
      break;
    }

    case "create-domain": {
      const [, project, service, host] = positional;
      const port = Number(flags.port ?? 3000);
      out = await runProcedure(
        "createDomain",
        {
          certificateResolver: flags.resolver ?? "",
          host,
          https: true,
          id: "",
          middlewares: [],
          path: "/",
          wildcard: false,
          destinationType: "service",
          serviceDestination: {
            path: "/",
            port,
            projectName: project,
            protocol: "http",
            serviceName: service,
          },
        },
        "mutation",
      );
      break;
    }

    case "set-env": {
      const [, project, service] = positional;
      out = await runProcedure(
        "updateAppEnv",
        { projectName: project, serviceName: service, env: envFromFlags(flags) ?? "" },
        "destructive",
      );
      break;
    }

    case "deploy": {
      const [, project, service] = positional;
      out = await runProcedure(
        "deployAppService",
        {
          projectName: project,
          serviceName: service,
          forceRebuild: Boolean(flags.force),
        },
        "destructive",
      );
      break;
    }

    case "inspect": {
      const [, project, service] = positional;
      const type = flags.type ?? "app";
      const procedure =
        type === "postgres"
          ? "inspectPostgresService"
          : type === "redis"
            ? "inspectRedisService"
            : "inspectAppService";
      out = await runProcedure(
        procedure,
        { projectName: project, serviceName: service },
        "query",
      );
      break;
    }

    case "raw": {
      const [, risk, procedure, json] = positional;
      out = await runProcedure(procedure, JSON.parse(json ?? "{}"), risk);
      break;
    }

    default:
      throw new Error(`Perintah tidak dikenal: ${command}\n\n${HELP}`);
  }

  console.log(typeof out === "string" ? out : JSON.stringify(out, null, 2));
}

main().catch((error) => {
  console.error(`ERROR: ${error.message}`);
  process.exitCode = 1;
});
