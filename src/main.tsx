import React from "react";
import ReactDOM from "react-dom/client";
import { Home, Search } from "./pages";
import { PrivyProvider } from "@privy-io/react-auth";
import "./index.css";
import { createBrowserRouter, RouterProvider } from "react-router-dom";

const router = createBrowserRouter([
  {
    path: "/",
    element: <Home />,
  },
  {
    path: "/search",
    element: <Search />,
  },
]);

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <PrivyProvider
      appId="clzhp6hio00yjd3qba4uhh4ho"
      config={{
        // Display email and wallet as login methods
        loginMethods: ["email", "wallet", "farcaster", "telegram"],
        // Customize Privy's appearance in your app
        appearance: {
          theme: "#fff",
          accentColor: "#fff",
          logo: "https://pub-dc971f65d0aa41d18c1839f8ab426dcb.r2.dev/privy.png",
          walletList: [
            "coinbase_wallet",
            "metamask",
            "rainbow",
            "rabby_wallet",
          ],
        },
        // Create embedded wallets for users who don't have a wallet
        embeddedWallets: {
          createOnLogin: "users-without-wallets",
        },
        externalWallets: {
          coinbaseWallet: {
            // Valid connection options include 'eoaOnly' (default), 'smartWalletOnly', or 'all'
            connectionOptions: "smartWalletOnly",
          },
        },
      }}
    >
      <RouterProvider router={router} />
    </PrivyProvider>
  </React.StrictMode>
);
