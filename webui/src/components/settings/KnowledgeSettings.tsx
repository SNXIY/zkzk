import { useCallback, useEffect, useRef, useState } from "react";
import { FileText, Loader2, RefreshCw, Trash2, Upload } from "lucide-react";
import { useTranslation } from "react-i18next";

import { Button } from "@/components/ui/button";
import {
  deleteKnowledgeDocument,
  fetchKnowledge,
  reindexKnowledge,
} from "@/lib/api";
import type { KnowledgeBackendStatus, KnowledgeDocument } from "@/lib/types";
import { useClient } from "@/providers/ClientProvider";

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function KnowledgeSettings({ token }: { token: string }) {
  const { t } = useTranslation();
  const { client } = useClient();
  const inputRef = useRef<HTMLInputElement>(null);
  const [docs, setDocs] = useState<KnowledgeDocument[]>([]);
  const [maxUploadMb, setMaxUploadMb] = useState(32);
  const [backend, setBackend] = useState<KnowledgeBackendStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const payload = await fetchKnowledge(token);
      setDocs(payload.documents);
      setMaxUploadMb(payload.max_upload_mb);
      setBackend(payload.backend ?? null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setDocs([]);
      setBackend(null);
    } finally {
      setLoading(false);
    }
  }, [token]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const onUpload = async (file: File | null) => {
    if (!file || !client) return;
    setBusy(true);
    setError(null);
    setMessage(null);
    try {
      const document = await client.uploadKnowledgeFile(file);
      setMessage(
        document.status === "queued" || document.status === "indexing"
          ? t("settings.knowledge.queued", {
              name: document.filename,
              status: document.status,
              defaultValue: "Queued {{name}} ({{status}}) — indexing in background",
            })
          : t("settings.knowledge.uploaded", {
              name: document.filename,
              chunks: document.chunk_count,
              defaultValue: "Indexed {{name}} ({{chunks}} chunks)",
            }),
      );
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
      if (inputRef.current) inputRef.current.value = "";
    }
  };

  const onDelete = async (doc: KnowledgeDocument) => {
    setBusy(true);
    setError(null);
    setMessage(null);
    try {
      await deleteKnowledgeDocument(token, doc.id);
      setMessage(
        t("settings.knowledge.deleted", {
          name: doc.filename,
          defaultValue: "Deleted {{name}}",
        }),
      );
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  const onReindex = async () => {
    setBusy(true);
    setError(null);
    setMessage(null);
    try {
      const result = await reindexKnowledge(token);
      setMessage(
        t("settings.knowledge.reindexed", {
          count: result.reindexed,
          total: result.total,
          defaultValue: "Reindexed {{count}} / {{total}} documents",
        }),
      );
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="space-y-7">
      <section className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
        <p className="max-w-[680px] text-[13px] leading-5 text-muted-foreground">
          {t("settings.knowledge.description", {
            defaultValue:
              "Enterprise RAG: structure-aware chunking + DashScope embeddings + DashVector. The agent searches with kb_search.",
          })}
        </p>
        <div className="flex flex-wrap items-center gap-2">
          <input
            ref={inputRef}
            type="file"
            className="hidden"
            accept=".pdf,.txt,.md,.markdown,.html,.htm,.csv,.log"
            onChange={(e) => void onUpload(e.target.files?.[0] ?? null)}
          />
          <Button
            type="button"
            size="sm"
            disabled={busy || !client}
            onClick={() => inputRef.current?.click()}
          >
            {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Upload className="h-4 w-4" />}
            <span className="ml-1.5">
              {t("settings.knowledge.upload", { defaultValue: "Upload" })}
            </span>
          </Button>
          <Button type="button" size="sm" variant="outline" disabled={busy} onClick={() => void onReindex()}>
            <RefreshCw className="h-4 w-4" />
            <span className="ml-1.5">
              {t("settings.knowledge.reindex", { defaultValue: "Reindex" })}
            </span>
          </Button>
        </div>
      </section>

      <p className="text-[12px] text-muted-foreground">
        {t("settings.knowledge.limit", {
          mb: maxUploadMb,
          defaultValue: "Max upload size {{mb}} MB · PDF / Markdown / text / HTML",
        })}
      </p>

      {backend ? (
        <div
          className={
            backend.health === "ok"
              ? "rounded-[10px] border border-border/50 bg-muted/35 px-3 py-2 text-[12px] text-foreground/80"
              : "rounded-[10px] border border-amber-500/35 bg-amber-500/10 px-3 py-2 text-[12px] text-foreground/85"
          }
        >
          <div className="font-medium">
            {t("settings.knowledge.backendStatus", {
              defaultValue: "Retrieval backend",
            })}
            {": "}
            {backend.effective_mode || backend.search_mode || "lexical"}
            {backend.vector_enabled
              ? " · DashVector on"
              : " · DashVector off (FTS only)"}
          </div>
          {backend.status_message ? (
            <div className="mt-0.5 text-muted-foreground">{backend.status_message}</div>
          ) : null}
          {backend.async_index ? (
            <div className="mt-0.5 text-muted-foreground">
              {t("settings.knowledge.asyncPipeline", {
                storage: backend.storage?.backend || "local",
                threshold: backend.async_threshold_mb ?? 5,
                kafka: backend.kafka?.enabled ? "on" : "off",
                defaultValue:
                  "Async index · storage={{storage}} · threshold={{threshold}}MB · Kafka={{kafka}}",
              })}
            </div>
          ) : null}
          {backend.direct_upload ? (
            <div className="mt-0.5 text-muted-foreground">
              {t("settings.knowledge.directUpload", {
                defaultValue: "Direct upload · browser → MinIO (presigned / multipart)",
              })}
            </div>
          ) : null}
        </div>
      ) : null}

      {message ? <p className="text-[13px] text-foreground/80">{message}</p> : null}
      {error ? <p className="text-[13px] text-destructive">{error}</p> : null}

      <section>
        <div className="flex items-center justify-between border-b border-border/45 pb-3">
          <h2 className="mb-2 px-1 text-[13px] font-semibold tracking-[-0.01em] text-foreground/85">
            {t("settings.knowledge.documents", { defaultValue: "Documents" })}
          </h2>
          <span className="rounded-full bg-muted px-2.5 py-1 text-[12px] font-medium text-muted-foreground">
            {docs.length}
          </span>
        </div>

        {loading ? (
          <div className="flex items-center gap-2 px-3 py-10 text-sm text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin" />
            {t("settings.knowledge.loading", { defaultValue: "Loading…" })}
          </div>
        ) : docs.length === 0 ? (
          <div className="px-3 py-12 text-center text-sm text-muted-foreground">
            {t("settings.knowledge.empty", {
              defaultValue: "No documents yet. Upload a PDF or Markdown file to start.",
            })}
          </div>
        ) : (
          <ul className="divide-y divide-border/40">
            {docs.map((doc) => (
              <li key={doc.id} className="flex items-start gap-3 px-2 py-3">
                <div className="mt-0.5 flex h-10 w-10 shrink-0 items-center justify-center rounded-[12px] bg-muted/70 text-muted-foreground">
                  <FileText className="h-4 w-4" />
                </div>
                <div className="min-w-0 flex-1">
                  <div className="truncate text-[13px] font-medium text-foreground">
                    {doc.filename}
                  </div>
                  <div className="mt-0.5 text-[12px] text-muted-foreground">
                    {doc.status}
                    {" · "}
                    {formatBytes(doc.bytes)}
                    {" · "}
                    {t("settings.knowledge.chunks", {
                      count: doc.chunk_count,
                      defaultValue: "{{count}} chunks",
                    })}
                    {doc.error ? ` · ${doc.error}` : ""}
                  </div>
                </div>
                <Button
                  type="button"
                  size="icon"
                  variant="ghost"
                  disabled={busy}
                  aria-label={t("settings.knowledge.delete", {
                    name: doc.filename,
                    defaultValue: "Delete {{name}}",
                  })}
                  onClick={() => void onDelete(doc)}
                >
                  <Trash2 className="h-4 w-4" />
                </Button>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}
