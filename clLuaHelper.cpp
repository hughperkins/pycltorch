#include "clLuaHelper.h"

#include "lua.h"
#include "luaT.h"
#include "THTensor.h"

#include <iostream>
using namespace std;

void clDumpStack(lua_State *L) {
    cout << "luatop " << lua_gettop(L) << endl;
    if(lua_gettop(L) >= 1 ) cout << "-1 type " << lua_typename(L, lua_type(L, -1)) << endl;
    if(lua_gettop(L) >= 2 ) cout << "-2 type " << lua_typename(L, lua_type(L, -2)) << endl;
    if(lua_gettop(L) >= 3 ) cout << "-3 type " << lua_typename(L, lua_type(L, -3)) << endl;
}
void clPopAsSelf(lua_State *L, void *instanceKey) {
    lua_pushlightuserdata(L, instanceKey);
    lua_insert(L, -2);
    lua_settable(L, LUA_REGISTRYINDEX);
}
void clDeleteSelf(lua_State *L, void *instanceKey) {
    lua_pushlightuserdata(L, instanceKey);
    lua_pushnil(L);
    lua_settable(L, LUA_REGISTRYINDEX);
}
void clPushSelf(lua_State *L, void *instanceKey) {
    lua_pushlightuserdata(L, instanceKey);
    lua_gettable(L, LUA_REGISTRYINDEX);
}
void clGetInstanceField(lua_State *L, void *instanceKey, const char *name) {
    lua_pushlightuserdata(L, instanceKey);
    lua_gettable(L, LUA_REGISTRYINDEX);
    lua_getfield(L, -1, name);
    lua_remove(L, -2);
}
THFloatTensor *clPopFloatTensor(lua_State *L) {
    void **pTensor = (void **)lua_touserdata(L, -1);
    THFloatTensor *tensor = (THFloatTensor *)(*pTensor);
    lua_remove(L, -1);
    return tensor;
}
const char * clPopString(lua_State *L) {
    const char *res = lua_tostring(L, -1);
    lua_remove(L, -1);
    return res;
}
float clPopFloat(lua_State *L) {
    float res = lua_tonumber(L, -1);
    lua_remove(L, -1);
    return res;
}
void clPushFloatTensor(lua_State *L, THFloatTensor *tensor) {
    THFloatTensor_retain(tensor);
    luaT_pushudata(L, tensor, "torch.FloatTensor");
}
void clGetGlobal(lua_State *L, const char *name1) {
    lua_getglobal(L, name1);
}
void clGetGlobal(lua_State *L, const char *name1, const char *name2) {
    lua_getglobal(L, name1);
    lua_getfield(L, -1, name2);
    lua_remove(L, -2);
}
void clGetGlobal(lua_State *L, const char *name1, const char *name2, const char *name3) {
    lua_getglobal(L, name1);
    lua_getfield(L, -1, name2);
    lua_remove(L, -2);
    lua_getfield(L, -1, name3);
    lua_remove(L, -2);
}

