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

#include "luaT.h"
#include "THTensor.h"
#include "THStorage.h"
//#include "clLuaHelper.h"
#include "LuaHelper.h"
#include "clnnWrapper.h"
#include "THClTensor.h"

using namespace std;

//#pragma message("compiling clnnWrapper")
THClState *getState(lua_State *L) {
    pushGlobal(L, "cltorch", "_state");
    void *state = lua_touserdata(L, -1);
//    cout << "state: " << (long)state << endl;
    lua_remove(L, -1);
    return (THClState *)state;
}
THClTensor *popClTensor(lua_State *L) {
    void **pTensor = (void **)lua_touserdata(L, -1);
    THClTensor *tensor = (THClTensor *)(*pTensor);
    lua_remove(L, -1);
    return tensor;
}
void pushClTensor(THClState *state, lua_State *L, THClTensor *tensor) {
    THClTensor_retain(state, tensor);
    luaT_pushudata(L, tensor, "torch.ClTensor");    
}

