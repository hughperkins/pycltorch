#pragma once

// this is just a copy of LuaHelper.h, but I havent thought of a good way of sharing them across PyTorch and ClPyTorch yet
// I'll figure that out later...

struct lua_State;
struct THFloatTensor;

void clDumpStack(lua_State *L);

void clGetGlobal(lua_State *L, const char *name1);
void clGetGlobal(lua_State *L, const char *name1, const char *name2);
void clGetGlobal(lua_State *L, const char *name1, const char *name2, const char *name3);

void clPopAsSelf(lua_State *L, void *instanceKey);
void clDeleteSelf(lua_State *L, void *instanceKey);
void clPushSelf(lua_State *L, void *instanceKey);
void clGetInstanceField(lua_State *L, void *instanceKey, const char *name);
THFloatTensor *clPopFloatTensor(lua_State *L);
const char * clPopString(lua_State *L);
float clPopFloat(lua_State *L);
void clPushFloatTensor(lua_State *L, THFloatTensor *tensor);

