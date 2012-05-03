/*
 * Copyright (c) 2011, Matthieu FAESSEL and ARMINES
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the University of California, Berkeley nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS AND CONTRIBUTORS BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


#include "DTest.h"

void TestSuite::add(TestCase *f)
{
    funcList.push_back(f);
}

RES_T TestSuite::run()
{
    RES_T retVal = RES_OK;
    
    list<TestCase*>::iterator f;
    int totTestsNbr = 0;
    int curTestNbr = 1;
    int nPassed = 0;
    int nFailed = 0;
    int tElapsedTot = 0;
    
    for (f=funcList.begin();f!=funcList.end();f++)
      totTestsNbr++;
    
    for (f=funcList.begin();f!=funcList.end();f++)
    {
	TestCase *tc = *f;
	std::stringstream ss;
	tc->init();
	tc->retVal = RES_OK;
	tc->outStream = &ss;
	
	cout << "Test #" << (curTestNbr++) << "/" << totTestsNbr << ": " << (*f)->name << "\t";
	
	int tStart = clock();
	tc->run();
	int tElapsed =  clock() - tStart;
	
	if (tc->retVal==RES_OK)
	{
	    cout << "Passed\t" << displayTime(tElapsed) << endl;
	    nPassed += 1;
	}
	else
	{
	    retVal = RES_ERR;
	    cout << "Failed:" << endl;
	    cout << ss.str();
	    nFailed += 1;
	}
	(*f)->end();
    }
    return retVal;
}
