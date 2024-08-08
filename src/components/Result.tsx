import { Answer } from "./Answer";
import { Related } from "./Related";
import { Sources } from "./Sources";
import { Relate } from "../interfaces/relate";
import { Source } from "../interfaces/source";
import { parseStreaming } from "../utils/parse-streaming";
import { FC, useEffect, useState } from "react";

export const Result: FC<{ query: string; rid: string }> = ({ query, rid }) => {
  const [sources, setSources] = useState<Source[]>([]);
  const [markdown, setMarkdown] = useState<string>("");
  const [relates, setRelates] = useState<Relate[] | null>(null);
  const [error, setError] = useState<number | null>(null);
  useEffect(() => {
    const controller = new AbortController();
    void parseStreaming(
      controller,
      query,
      rid,
      setSources,
      setMarkdown,
      setRelates,
      setError
    );
    return () => {
      controller.abort();
    };
  }, [query]);
  return (
    <div className="flex flex-col gap-8">
      <Answer markdown={markdown} sources={sources} />
      <Sources sources={sources} />
      <Related relates={relates} />
      {error && (
        <div className="absolute inset-4 flex items-center justify-center  backdrop-blur-sm">
          <div className="p-4 shadow-2xl rounded text-blue-500 font-medium flex gap-4">
            {error === 429
              ? "Sorry, you have made too many requests recently, try again later."
              : "Sorry, we might be overloaded, try again later."}
          </div>
        </div>
      )}
    </div>
  );
};
