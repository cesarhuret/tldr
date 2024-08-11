import React, { FC } from "react";

export const Logo: FC = () => {
  return (
    <div className="flex gap-4 items-center justify-center cursor-default select-none relative">
      <img src="/icons/book.svg" className="h-16" alt="TLDR AI" />
      <div className="text-center font-medium text-4xl md:text-6xl text-white relative text-nowrap">
        TLDR AI
      </div>
      <div className="transform scale-75 origin-left border items-center rounded-lg bg-transparent px-2 py-1 text-xs font-medium text-white">
        beta
      </div>
    </div>
  );
};
