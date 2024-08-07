import React, { FC, useState } from "react";

export const Search: FC<{ _prompt?: string }> = ({
  _prompt,
}: {
  _prompt?: string;
}) => {
  const [prompt, setPrompt] = useState(_prompt);
  // const router = useRouter();
  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        if (prompt) {
          setPrompt("");
          // router.push(getSearchUrl(encodeURIComponent(value), nanoid()));
        }
      }}
    >
      <label
        className="relative flex items-center justify-center border ring-zinc-300/20 py-2 px-2 rounded-lg gap-2 border-[#252525] shadow-[#0d0d0d] shadow-xl transition-all focus:scale-95"
        htmlFor="search-bar"
      >
        <input
          id="search-bar"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          autoFocus
          placeholder="Ask TLDR AI anything..."
          className="px-2 pr-6 w-full rounded-md flex-1 outline-none bg-transparent text-white"
        />
        <button
          type="submit"
          className="w-auto py-1 px-2 text-white active:scale-95 overflow-hidden relative rounded-lg"
        >
          <img src="/icons/search.svg" alt="search" className="h-4 w-4" />
        </button>
      </label>
    </form>
  );
};
