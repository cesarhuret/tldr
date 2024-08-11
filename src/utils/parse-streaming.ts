import { Action } from "../interfaces/action";
import { Relate } from "../interfaces/relate";
import { Source } from "../interfaces/source";
import { fetchStream } from "./fetch-stream";
const LLM_SPLIT = "__LLM_RESPONSE__";
const RELATED_SPLIT = "__RELATED_QUESTIONS__";

export const parseStreaming = async (
  controller: AbortController,
  query: string,
  search_uuid: string,
  onSources: (value: Source[]) => void,
  onActions: (value: Action[]) => void,
  onMarkdown: (value: string) => void,
  onRelates: (value: Relate[]) => void,
  onError?: (status: number) => void
) => {
  const decoder = new TextDecoder();
  let uint8Array = new Uint8Array();
  let chunks = "";
  let sourcesEmitted = false;

  const url =
    process.env.NODE_ENV === "production"
      ? "https://api.librar.ie/query"
      : "http://localhost:8080/query";

  const response = await fetch(`http://localhost:8080/query`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "*./*",
    },
    signal: controller.signal,
    body: JSON.stringify({
      query,
      search_uuid,
    }),
  });
  if (response.status !== 200) {
    onError?.(response.status);
    return;
  }
  const markdownParse = (text: string) => {
    onMarkdown(
      text
        .replace(/\[\[([cC])itation/g, "[citation")
        .replace(/[cC]itation:(\d+)]]/g, "citation:$1]")
        .replace(/\[\[([cC]itation:\d+)]](?!])/g, `[$1]`)
        .replace(/\[[cC]itation:(\d+)]/g, "[citation]($1)")
    );
  };
  fetchStream(
    response,
    (chunk) => {
      uint8Array = new Uint8Array([...uint8Array, ...chunk]);
      chunks = decoder.decode(uint8Array, { stream: true });
      if (chunks.includes(LLM_SPLIT)) {
        console.log(chunks);

        const [sources, rest] = chunks.split(LLM_SPLIT);
        if (!sourcesEmitted) {
          try {
            const parsed = JSON.parse(sources);

            const actions = [];
            const _sources = [];

            for (const item of parsed) {
              if (!item?.url && item?.contract) {
                actions.push(item);
              } else if (item?.url) {
                _sources.push(item);
              }
            }

            console.log("ACTIONS: ", actions);

            onActions(actions);
            onSources(_sources);
          } catch (e) {
            onSources([]);
            onActions([]);
          }
        }
        sourcesEmitted = true;
        if (rest.includes(RELATED_SPLIT)) {
          const [md] = rest.split(RELATED_SPLIT);
          markdownParse(md);
        } else {
          markdownParse(rest);
        }
      }
    },
    () => {
      const [_, relates] = chunks.split(RELATED_SPLIT);
      try {
        onRelates(JSON.parse(relates));
      } catch (e) {
        onRelates([]);
      }
    }
  );
};
