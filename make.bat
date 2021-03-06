@echo off

REM # Copyright (c) 2014-2016, Intel Corporation All rights reserved.
REM # 
REM # Redistribution and use in source and binary forms, with or without 
REM # modification, are permitted provided that the following conditions are 
REM # met: 
REM # 
REM # 1. Redistributions of source code must retain the above copyright 
REM # notice, this list of conditions and the following disclaimer. 
REM #
REM # 2. Redistributions in binary form must reproduce the above copyright 
REM # notice, this list of conditions and the following disclaimer in the 
REM # documentation and/or other materials provided with the distribution. 
REM #
REM # 3. Neither the name of the copyright holder nor the names of its 
REM # contributors may be used to endorse or promote products derived from 
REM # this software without specific prior written permission. 
REM #
REM # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS 
REM # IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED 
REM # TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A 
REM # PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT 
REM # HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, 
REM # SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED 
REM # TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR 
REM # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF 
REM # LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING 
REM # NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS 
REM # SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

echo matrix_multiplication.c
icl -nologo -Qmic -I..\..\include -I"%MKLROOT%\include" -O2 -fPIC -shared -L"%MKLROOT%/lib/mic" -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread -lpthread -o matrix_multiplication.so matrix_multiplication.c
