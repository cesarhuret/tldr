export interface Action {
  contract: string;
  symbol: string;
  abi?: string;
  args: string;
  protocol?: any;
}
