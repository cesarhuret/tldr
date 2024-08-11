import { FC } from "react";
import { Action } from "../interfaces/action";
import { Wrapper } from "./Wrapper";
import { Skeleton } from "./Skeleton";
import { createWalletClient, custom, encodeFunctionData } from "viem";
import { truncate } from "../utils/truncate";
import { usePrivy, useWallets } from "@privy-io/react-auth";
import { mode } from "viem/chains";

const ActionItem: FC<{ source: Action; index: number }> = ({ source }) => {
  const { contract, symbol, abi, args, protocol } = source;

  const { ready, authenticated, login } = usePrivy();

  const { wallets } = useWallets();

  const interact = async () => {
    if (!authenticated) {
      login();
      return;
    }

    console.log(wallets);
    const wallet = wallets[0];
    const provider = await wallet.getEthereumProvider();

    const walletClient = createWalletClient({
      chain: mode,
      transport: custom(provider),
    });

    await walletClient.switchChain({ id: mode.id });

    const data = encodeFunctionData({
      abi: [abi],
      args: [1000000],
    });

    const hash = await walletClient.sendTransaction({
      account: wallet.address as `0x${string}`,
      to: contract as `0x${string}`,
      data: data as `0x${string}`,
    });

    console.log(hash);
  };

  return (
    <div
      className="relative text-xs py-3 px-3 bg-[#252525] hover:bg-[#353535] rounded-lg flex flex-row gap-2 items-center justify-start transition-all hover:cursor-pointer"
      key={contract}
    >
      <img
        className="w-7 h-min rounded"
        alt={protocol?.url}
        src={`https://www.google.com/s2/favicons?domain=${
          protocol?.url
        }&sz=${64}`}
      />
      <a onClick={interact} className="absolute inset-0"></a>
      <div className="flex flex-col items-start">
        <div className="font-medium text-ellipsis overflow-hidden whitespace-nowrap break-words">
          Interact With {symbol}
        </div>
        <div className="flex gap-2 items-center">
          <div className="flex-1 overflow-hidden">
            <div className="text-ellipsis whitespace-nowrap break-all text-zinc-400 overflow-hidden w-full">
              {truncate(contract, 13)}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export const Actions: FC<{ sources: Action[] }> = ({ sources }) => {
  return (
    <Wrapper
      title={"Actions"}
      content={
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
          {sources.length > 0 ? (
            sources.map((item, index) => (
              <ActionItem key={index} index={index} source={item} />
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
    />
  );
};
