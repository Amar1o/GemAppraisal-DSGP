import { createContext, useContext, useEffect, useState } from "react";
import { supabase } from "../supabaseClient"; // Ensure this path is correct

const AuthContext = createContext();

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true); // Add loading state

  useEffect(() => {
    const getSession = async () => {
      setLoading(true);
  
      // First, check local storage for an existing session
      const storedSession = localStorage.getItem("supabaseSession");
      if (storedSession) {
        const session = JSON.parse(storedSession);
        setUser(session.user);
      }
  
      // Then, verify with Supabase (to make sure session is still valid)
      const { data: { session } } = await supabase.auth.getSession();
      if (session) {
        setUser(session.user);
        localStorage.setItem("supabaseSession", JSON.stringify(session));
      } else {
        localStorage.removeItem("supabaseSession");
        setUser(null);
      }
  
      setLoading(false);
    };
  
    getSession();
  
    // Listen for auth changes
    const { data: listener } = supabase.auth.onAuthStateChange((_event, session) => {
      setUser(session?.user || null);
      if (session) {
        localStorage.setItem("supabaseSession", JSON.stringify(session));
      } else {
        localStorage.removeItem("supabaseSession");
      }
    });
  
    return () => {
      listener.subscription.unsubscribe();
    };
  }, []);  

  const login = async (email, password) => {
    const { data, error } = await supabase.auth.signInWithPassword({ email, password });
    if (error) throw error;
    setUser(data.user);
    localStorage.setItem("supabaseSession", JSON.stringify(data.session));
  };

  const logout = async () => {
    await supabase.auth.signOut();
    setUser(null);
    localStorage.removeItem("supabaseSession");
  };

  return (
    <AuthContext.Provider value={{ user, login, logout, loading }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => useContext(AuthContext);
