import { useState } from "react";
import { useAuth } from "./AuthContext";
import { useNavigate } from "react-router-dom";
import { supabase } from "../supabaseClient";
import { Link } from "react-router-dom";
function Navbar() {
    // State to track menu visibility
    const [isMenuOpen, setIsMenuOpen] = useState(false);
    const { user } = useAuth();
    const navigate = useNavigate();

    // Function to toggle menu visibility
    const toggleMenu = () => {
      setIsMenuOpen(!isMenuOpen);
    };

    // Logout function
    const handleLogout = async () => {
      await supabase.auth.signOut();
      navigate("/");
    };
    

    return (
        <header className='shadow-md bg-white font-[Open Sans] tracking-wide z-50 relative w-full'>
      {/* Logo and Button Section */}
      <section className="flex items-center lg:justify-center flex-wrap relative py-3 sm:px-10 px-4 border-gray-200 border-b lg:min-h-[80px] max-lg:min-h-[60px]">
        {/* Logo (Centered) */}
        <div className="md:absolute md:left-1/2 md:transform md:-translate-x-1/2">
          <a href="/">
            <img src="/logo.png" alt="logo" className="md:w-[170px] w-32" />
          </a>
        </div>

        {/* Right-Side Buttons */}
        <div className="ml-auto flex items-center gap-4">
          {user ? (
            <button
              onClick={handleLogout}
              className="bg-gray-600 hover:bg-gray-700 text-white font-bold py-2 px-3 rounded opacity-90 shadow-md flex items-center gap-2"
            >
              Logout
            </button>
          ) : (
            <div className="flex gap-3">
              <button
                onClick={() => navigate("/login")}
                className="bg-blue-500 hover:bg-blue-600 text-white font-bold py-2 px-3 rounded shadow-md"
              >
                Login
              </button>
              <button
                onClick={() => navigate("/signup")}
                className="bg-green-500 hover:bg-green-600 text-white font-bold py-2 px-3 rounded shadow-md"
              >
                Sign Up
              </button>
            </div>
          )}
        </div>
      </section>


      {/* Navbar Menu */}
      <div className="flex flex-wrap py-3.5 px-10 overflow-x-auto">
        {/* Menu Items */}
        <div
          className={`w-full lg:flex lg:justify-center lg:gap-x-10 ${
            isMenuOpen
              ? "fixed bg-white w-3/4 min-w-[300px] top-0 left-0 p-6 h-full shadow-lg transition-all duration-300"
              : "hidden lg:block"
          }`}
        >
          {/* Close Button for Mobile */}
          <button
            className="lg:hidden absolute top-4 right-4 bg-gray-200 w-9 h-9 flex items-center justify-center rounded-full"
            onClick={toggleMenu}
          >
            ✖
          </button>

          {/* Menu Links */}
          <ul className="lg:flex lg:justify-center lg:gap-x-10 max-lg:space-y-3 max-lg:fixed max-lg:bg-white max-lg:w-1/2 max-lg:min-w-[300px] max-lg:top-0 max-lg:left-0 max-lg:p-6 max-lg:h-full max-lg:shadow-md max-lg:overflow-auto z-50">
            {["Home", "Team", "About", "Contact"].map((item) => (
              <li key={item} className="max-lg:border-b max-lg:py-3">
                <a
                  href="/#{item}"
                  className="hover:text-gray-700 text-gray-500 font-bold text-l block"
                >
                  {item}
                </a>
              </li>
            ))}
            {/* Conditionally render Price Calculator link based on user */}
            {user && (
              <li className="max-lg:border-b max-lg:py-3">
                <Link
                  to="/price-calculator"
                  className="hover:text-gray-700 text-gray-500 font-bold text-l block"
                >
                  Calculator
                </Link>
              </li>
            )}
          </ul>
        </div>

        {/* Hamburger Menu Button (Mobile) */}
        <div className="flex ml-auto lg:hidden">
          <button onClick={toggleMenu}>
            <svg className="w-7 h-7" fill="#000" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg">
              <path
                fillRule="evenodd"
                d="M3 5a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zM3 10a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zM3 15a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1z"
                clipRule="evenodd"
              ></path>
            </svg>
          </button>
        </div>
      </div>
    </header>
    );
}

export default Navbar