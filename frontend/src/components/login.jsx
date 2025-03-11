import React, { useState } from 'react';
import { supabase } from '../supabaseClient';
import { useNavigate } from 'react-router-dom';  // Import useNavigate

const Login = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const navigate = useNavigate();  // Initialize navigation

  const handleLogin = async (e) => {
    e.preventDefault();
    const { data, error } = await supabase.auth.signInWithPassword({ email, password });
    if (error) {
      setError(error.message);
    } else {
      setSuccess('Login successful!');
      setError('');
      setTimeout(() => {
        navigate('/price-calculator');  // Redirect user to dashboard
      }, 1000);
    }
  };

  return (
    <div className="relative min-h-screen flex items-center justify-center bg-cover bg-center before:absolute before:w-full before:h-full before:inset-0 before:bg-black before:opacity-50 before:z-10" style={{ backgroundImage: "url('/login_image.jpg')" }}>
    <div className="z-40 flex min-h-full flex-1 flex-col justify-center px-6 py-12 lg:px-8 bg-white shadow-[0_4px_12px_-5px_rgba(0,0,0,0.4)] w-full max-w-sm rounded-lg font-[Open Sans] overflow-hidden mx-auto opacity-90">
      <div className="sm:mx-auto sm:w-full sm:max-w-sm">
        <img
              alt="It Saphyre"
              src="/logo.png"
              className="mx-auto h-12 w-auto"
            />
        <h2 className="mt-10 text-center text-2xl/9 tracking-tight font-bold text-gray-700">
            Sign in to your account
        </h2>
        </div>
        <div className="mt-10 sm:mx-auto sm:w-full sm:max-w-sm">
          {error && <div className="alert alert-error">{error}</div>}
          {success && <div className="alert alert-success">{success}</div>}
          <form onSubmit={handleLogin} className="space-y-6">
          <div>
              <label htmlFor="email" className="block text-sm/6 font-medium text-gray-700">
                Email address
              </label>
              <div className="mt-2">
              <input
                type="email"
                id="email"
                name="email"
                placeholder="Email"
                className="block w-full rounded-md bg-white px-3 py-1.5 text-base text-gray-700 outline-1 -outline-offset-1 outline-gray-300 placeholder:text-gray-400 focus:outline-2 focus:-outline-offset-2 focus:outline-cyan-600 sm:text-sm/6"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
              />
              </div>
            </div>
            <div>
              <div className="flex items-center justify-between">
                <label htmlFor="password" className="block text-sm/6 font-medium text-gray-900">
                  Password
                </label>
                <div className="text-sm">
                  <a href="#" className="font-semibold text-cyan-600 hover:text-cyan-500">
                    Forgot password?
                  </a>
                </div>
              </div>
              <div className="mt-2">
              <input
                type="password"
                placeholder="Password"
                className="block w-full rounded-md bg-white px-3 py-1.5 text-base text-gray-900 outline-1 -outline-offset-1 outline-gray-300 placeholder:text-gray-400 focus:outline-2 focus:-outline-offset-2 focus:outline-cyan-600 sm:text-sm/6"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
              />
              </div>
            </div>
            <div>
              <button
                type="submit"
                className="flex w-full justify-center rounded-md bg-cyan-600 px-3 py-1.5 text-sm/6 font-semibold text-white shadow-xs hover:bg-cyan-500 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-cyan-600"
                >
                Sign in
              </button>
            </div>
          </form>
          <p className="mt-10 text-center text-sm/6 text-gray-500">
            Not a member?{' '}
            <a href="/signup" className="font-semibold text-cyan-600 hover:text-cyan-500">
              Sign up for free
            </a>
          </p>
        </div>
      </div>
    </div>
  );
};

export default Login;
