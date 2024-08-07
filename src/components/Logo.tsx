import React, { FC } from "react";

export const Logo: FC = () => {
  return (
    <div className="flex gap-4 items-center justify-center cursor-default select-none relative">
      <div className="h-10 w-10">
        <img src="/icons/book.svg" alt="TLDR AI" />
      </div>
      <div className="text-center font-medium text-2xl md:text-3xl text-white relative text-nowrap">
        TLDR AI
      </div>
      <div className="transform scale-75 origin-left border items-center rounded-lg bg-transparent px-2 py-1 text-xs font-medium text-white">
        beta
      </div>
    </div>
  );
};
