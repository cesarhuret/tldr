import { nanoid } from "nanoid";
import { FC, useMemo } from "react";
import { getSearchUrl } from "../utils/get-search-url";

export const PresetQuery: FC<{ query: string }> = ({ query }) => {
  const rid = useMemo(() => nanoid(), [query]);

  return (
    <a
      title={query}
      href={getSearchUrl(query, rid)}
      className="border border-[#252525] text-ellipsis overflow-hidden text-nowrap items-center rounded-lg bg-transparent transition-all hover:text-[#e0e0e0] px-2 py-1 text-xs font-medium text-[#c0c0c0]"
    >
      {query}
    </a>
  );
};
