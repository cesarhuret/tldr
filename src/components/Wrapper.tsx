import { FC, ReactNode } from "react";

export const Wrapper: FC<{
  title: ReactNode;
  content: ReactNode;
}> = ({ title, content }) => {
  return (
    <div className="flex flex-col gap-4 w-full">
      <p className="gap-2 w-min text-gray-200 drop-shadow-xl shadow-[#000]">
        {title}
      </p>
      {content}
    </div>
  );
};
