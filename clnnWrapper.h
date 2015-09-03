#pragma once

#include <iostream>
#include <string>

class THFloatTensor;
class THFloatStorage;
class THClState;
struct lua_State;
class THClTensor;

THClState *getState(lua_State *L);
THClTensor *popClTensor(lua_State *L);
void pushClTensor(THClState *state, lua_State *L, THClTensor *tensor);

