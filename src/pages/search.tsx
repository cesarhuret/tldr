import { useNavigate, useSearchParams } from "react-router-dom";
import { Title } from "../components/Title";
import { Result } from "../components/Result";
import { Search as SearchComponent } from "../components/Search";

export const Search = () => {
  let [searchParams] = useSearchParams();
  const query = decodeURIComponent(searchParams.get("q") || "");
  const rid = decodeURIComponent(searchParams.get("rid") || "");

  const navigate = useNavigate();
  return (
    <div className="absolute inset-0">
      <div
        className="absolute inset-0"
        onClick={() => {
          navigate("/");
        }}
      />

      <div className="mx-auto max-w-3xl absolute inset-4 md:inset-12 animate-fade shadow-[#090909] shadow-xl">
        <div className="h-20 pointer-events-none rounded-t-2xl w-full backdrop-filter absolute top-0 [mask-image:linear-gradient(to_bottom,white,transparent)]"></div>
        <div className="px-4 md:px-8 pt-6 pb-24 rounded-2xl border shadow-[#151515] shadow-inner drop-shadow-2xl border-[#252525] h-full overflow-auto">
          <Title query={query} />
          <Result key={rid} query={query} rid={rid} />
        </div>
        <div className="h-80 pointer-events-none w-full rounded-b-2xl backdrop-filter absolute bottom-0 bg-gradient-to-b from-transparent to-transparent [mask-image:linear-gradient(to_top,white,transparent)]"></div>
        <div className="absolute z-10 flex items-center justify-center bottom-6 px-4 md:px-8 w-full">
          <div className="w-full">
            <SearchComponent />
          </div>
        </div>
      </div>
    </div>
  );
};
