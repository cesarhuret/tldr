import { Skeleton } from "./Skeleton";
import { Wrapper } from "./Wrapper";
import { Source } from "../interfaces/source";
import { FC } from "react";

const SourceItem: FC<{ source: Source; index: number }> = ({ source }) => {
  const { id, name, url } = source;
  const domain = new URL(url).hostname;
  return (
    <div
      className="relative text-xs py-3 px-3 bg-[#252525] hover:bg-[#353535] rounded-lg flex flex-row gap-2 items-center justify-start transition-all"
      key={id}
    >
      <img
        className="w-7 h-min rounded"
        alt={domain}
        src={`https://www.google.com/s2/favicons?domain=${domain}&sz=${64}`}
      />
      <a href={url} target="_blank" className="absolute inset-0"></a>
      <div className="flex flex-col items-start">
        <div className="font-medium text-ellipsis overflow-hidden whitespace-nowrap break-words">
          {name}
        </div>
        <div className="flex gap-2 items-center">
          <div className="flex-1 overflow-hidden">
            <div className="text-ellipsis whitespace-nowrap break-all text-zinc-400 overflow-hidden w-full">
              {domain}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export const Sources: FC<{ sources: Source[] }> = ({ sources }) => {
  return (
    <Wrapper
      title={"Sources"}
      content={
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
          {sources.length > 0 ? (
            sources.map((item, index) => (
              <SourceItem key={item.id} index={index} source={item} />
            ))
          ) : (
            <>
              <Skeleton className="max-w-sm h-16 bg-[#222]" />
              <Skeleton className="max-w-sm h-16 bg-[#222]" />
              <Skeleton className="max-w-sm h-16 bg-[#222]" />
              <Skeleton className="max-w-sm h-16 bg-[#222]" />
            </>
          )}
        </div>
      }
    ></Wrapper>
  );
};
