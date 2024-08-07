import { Search } from "../components/Search";
import { Logo } from "../components/Logo";
import { PresetQuery } from "../components/PresetQuery";
import { Footer } from "../components/Footer";

export const Home = () => {
  return (
    <div className="absolute inset-0 min-h-[500px] flex items-center justify-center">
      <div className="relative flex flex-col gap-8 px-4 -mt-24">
        <Logo />
        <div className="flex flex-col w-full gap-3">
          <Search />
          <div className="flex gap-2 flex-wrap justify-center">
            <PresetQuery query="How do I bridge to Optimism?" />
            <PresetQuery query="What are the hottest yields on Base right now?" />
          </div>
        </div>
        <Footer />
      </div>
    </div>
  );
};
