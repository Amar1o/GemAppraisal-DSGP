import { useState } from "react";
import { useAuth } from "./AuthContext";
import { useNavigate, useLocation } from "react-router-dom";
import { supabase } from "../supabaseClient";
import { Link } from "react-router-dom";

function Navbar() {
    const [isMenuOpen, setIsMenuOpen] = useState(false);
    const { user } = useAuth();
    const navigate = useNavigate();
    const location = useLocation(); // Get current page location

    // Function to toggle menu visibility
    const toggleMenu = () => {
        setIsMenuOpen(!isMenuOpen);
    };

    // Logout function
    const handleLogout = async () => {
        await supabase.auth.signOut();
        navigate("/login");
    };

    // Function to navigate to a section (works from any page)
    const scrollToSection = (id) => {
        if (location.pathname !== "/") {
            // If not on Home, go to Home first
            navigate("/", { state: { scrollTo: id } });
        } else {
            // If already on Home, scroll directly
            setTimeout(() => {
                const section = document.getElementById(id);
                if (section) {
                    section.scrollIntoView({ behavior: "smooth" });
                }
            }, 100); // Delay for smooth transition
        }
        setIsMenuOpen(false); // Close menu on mobile after clicking
    };

    return (
        <header className='shadow-md  font-[Inter] tracking-wide z-50 w-full sticky top-0  h-full bg-transparent bg-clip-padding backdrop-filter backdrop-blur-md'>
            <nav className=" border-gray-200 px-4 lg:px-6 py-2.5 dark:bg-gray-800">
            <div className="flex flex-wrap justify-between items-center mx-auto ">
            <a href="/" className="flex">
              <img src="/ItSaphyre.png" alt="logo" className="mr-3 w-40" />
            </a>
            <div className="flex items-center lg:order-2">
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
                                className="bg-cyan-500 hover:bg-cyan-600 text-white font-bold py-2 px-3 rounded-xl shadow-2xl "
                            >
                                Login
                            </button>
                            <button
                                onClick={() => navigate("/signup")}
                                className="bg-white hover:bg-cyan-600 text-cyan-600 hover:text-white font-bold py-2 px-3 rounded-xl shadow-2xl border-2 border-cyan-600"
                            >
                                Sign Up
                            </button>
                        </div>
                    )}
                <button data-collapse-toggle="mobile-menu-2" type="button" className="inline-flex items-center p-2 ml-1 text-sm text-gray-500 rounded-lg lg:hidden hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-gray-200 dark:text-gray-400 dark:hover:bg-gray-700 dark:focus:ring-gray-600" aria-controls="mobile-menu-2" aria-expanded="false">
                    <span className="sr-only">Open main menu</span>
                    <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg"><path fill-rule="evenodd" d="M3 5a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zM3 10a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zM3 15a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1z" clip-rule="evenodd"></path></svg>
                    <svg className="hidden w-6 h-6" fill="currentColor" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg"><path fill-rule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clip-rule="evenodd"></path></svg>
                </button>
            </div>
            <div className="hidden justify-between items-center w-full lg:flex lg:w-auto lg:order-1" id="mobile-menu-2">
                <ul className="flex flex-col mt-4 font-medium lg:flex-row lg:space-x-8 lg:mt-0">
                {["Home", "Team", "About", "Contact"].map((item) => (
                            <li key={item} className="max-lg:border-b max-lg:py-3">
                                <button
                                    onClick={() => scrollToSection(item.toLowerCase())} // Scrolls smoothly
                                    className="hover:text-gray-700 text-gray-500 font-bold text-l block"
                                >
                                    {item}
                                </button>
                            </li>
                        ))}
                        {/* Conditionally render Price Calculator link based on user */}
                        
                            <li className="max-lg:border-b max-lg:py-3">
                                <Link
                                    to="/price-calculator"
                                    className="hover:text-gray-700 text-gray-500 font-bold text-l block"
                                >
                                    Calculator
                                </Link>
                            </li>
                </ul>
            </div>
        </div>
    </nav>
  </header>
    );
}

export default Navbar;
