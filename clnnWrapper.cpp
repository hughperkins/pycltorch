extern "C" {
    #include "lua.h"
    #include "lauxlib.h"
    #include "lualib.h"
}

#ifndef _WIN32
    #include <dlfcn.h>
#endif

#include <iostream>
#include <stdexcept>

//#include "luaT.h"
#include "THTensor.h"
#include "THStorage.h"
//#include "clLuaHelper.h"
#include "LuaHelper.h"
#include "clnnWrapper.h"

using namespace std;

#pragma message("compiling clnnWrapper")
THClState *getState(lua_State *L) {
    pushGlobal(L, "cltorch", "_state");
    void *state = lua_touserdata(L, -1);
    cout << "state: " << (long)state << endl;
    lua_remove(L, -1);
    return (THClState *)state;
}

