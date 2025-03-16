import React, { useState } from 'react';
import { supabase } from '../supabaseClient.js';

const SignUp = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  const handleSignUp = async (e) => {
    e.preventDefault();
    const { error } = await supabase.auth.signUp({ email, password });
    if (error) {
      setError(error.message);
    } else {
      setSuccess('Sign-up successful! Check your email for confirmation.');
      setError('');
    }
  };

  return (
    <div className="relative min-h-screen flex items-center bg-slate-50 ">
    <div className="z-40 flex min-h-full flex-1 flex-col justify-center px-6 py-12 lg:px-8 bg-white shadow-[0_4px_12px_-5px_rgba(0,0,0,0.4)] w-full max-w-sm rounded-lg font-[Open Sans] overflow-hidden mx-auto opacity-90 ">
      <div className="sm:mx-auto sm:w-full sm:max-w-sm">
        <img
              alt="Your Company"
              src="/ItSaphyre.png"
              className="mx-auto h-12 w-auto"
            />
        <h2 className="mt-10 text-center text-2xl/9 font-bold tracking-tight text-gray-900">
            Sign up to create an account
        </h2>
        </div>
        <div className="mt-10 sm:mx-auto sm:w-full sm:max-w-sm">
          {error && <div className="alert alert-error">{error}</div>}
          {success && <div className="alert alert-success">{success}</div>}
          <form onSubmit={handleSignUp} className="space-y-6">
            <div>
                <label htmlFor="email" className="block text-sm/6 font-medium text-gray-900">
                  Email address
                </label>
                <div className="mt-2">
            <input
              type="email"
              placeholder="Email"
              className="block w-full rounded-md bg-white px-3 py-1.5 text-base text-gray-900 outline-1 -outline-offset-1 outline-gray-300 placeholder:text-gray-400 focus:outline-2 focus:-outline-offset-2 focus:outline-cyan-600 sm:text-sm/6"
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
              </div>
            <input
              type="password"
              placeholder="Password"
              className="block w-full rounded-md bg-white px-3 py-1.5 text-base text-gray-900 outline-1 -outline-offset-1 outline-gray-300 placeholder:text-gray-400 focus:outline-2 focus:-outline-offset-2 focus:outline-cyan-600 sm:text-sm/6"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
            </div>
            <div>
              <button
                type="submit"
                className="flex w-full justify-center rounded-md bg-cyan-600 px-3 py-1.5 text-sm/6 font-semibold text-white shadow-xs hover:bg-cyan-500 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-cyan-600"
                >
                Sign up
              </button>
            </div>
          </form>
          <p className="mt-10 text-center text-sm/6 text-gray-500">
            Already a member?{' '}
            <a href="/login" className="font-semibold text-cyan-600 hover:text-cyan-500">
              Sign in to your account
            </a>
          </p>
        </div>
        </div>
        <div className="md:w-1/2 md:min-h-screen bg-cover bg-center" style={{ backgroundImage: "url('/login_image.jpg')", clipPath: "polygon(20% 0%, 100% 0%, 100% 100%, 0% 100%)" }}></div>
        </div>
  );
};

export default SignUp;
