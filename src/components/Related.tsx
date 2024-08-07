import { PresetQuery } from "./PresetQuery";
import { Skeleton } from "./Skeleton";
import { Wrapper } from "./Wrapper";
import { Relate } from "../interfaces/relate";
import React, { FC } from "react";

export const Related: FC<{ relates: Relate[] | null }> = ({ relates }) => {
  return (
    <Wrapper
      title={<>Related</>}
      content={
        <div className="flex gap-2 flex-col">
          {relates !== null ? (
            relates.length > 0 ? (
              relates.map(({ question }) => (
                <PresetQuery key={question} query={question}></PresetQuery>
              ))
            ) : (
              <div className="text-sm">No related questions.</div>
            )
          ) : (
            <>
              <Skeleton className="w-full h-5 bg-[#222]"></Skeleton>
              <Skeleton className="w-full h-5 bg-[#222]"></Skeleton>
              <Skeleton className="w-full h-5 bg-[#222]"></Skeleton>
            </>
          )}
        </div>
      }
    ></Wrapper>
  );
};
