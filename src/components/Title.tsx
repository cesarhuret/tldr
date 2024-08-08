import { nanoid } from "nanoid";
import { useNavigate } from "react-router-dom";
import { getSearchUrl } from "../utils/get-search-url";

export const Title = ({ query }: { query: string }) => {
  const navigate = useNavigate();
  return (
    <div className="flex items-center pb-4 mb-6 border-b border-[#333] gap-4">
      <div
        className="flex-1 text-lg sm:text-xl text-ellipsis overflow-hidden whitespace-nowrap"
        title={query}
      >
        {query}
      </div>
      <div className="flex-none">
        <button
          onClick={() => {
            navigate(getSearchUrl(encodeURIComponent(query), nanoid()));
          }}
          type="button"
          className="rounded flex gap-2 items-center bg-transparent px-2 py-1 text-xs font-semibold text-gray-400 hover:bg-[#454545] transition-all"
        >
          Again?
        </button>
      </div>
    </div>
  );
};
