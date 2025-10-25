/*
#############################################################################################
# RocketLeague (1.0.10897.0) SDK
# Generated with the RLSDK Generator v1.0.0
# ========================================================================================= #
# File: OnlineSubsystemEOS_structs.hpp
# ========================================================================================= #
# Credits: imattas
# Links: www.github.com/imattas/RLSDK-Generator
#############################################################################################
*/
#pragma once

#ifdef _MSC_VER
	#pragma pack(push, 0x1)
#endif

/*
# ========================================================================================= #
# Structs
# ========================================================================================= #
*/

// ScriptStruct OnlineSubsystemEOS.OnlinePlayerInterfaceEOS.OnlineStatusMappingEOS
// 0x0018
struct FOnlineStatusMappingEOS
{
	int32_t                                            StatusId;                                      // 0x0000 (0x0004) [0x0000000000000000]               
	uint8_t                                           UnknownData00[0x4];                               		// 0x0004 (0x0004) MISSED OFFSET
	class FString                                      StatusString;                                  // 0x0008 (0x0010) [0x0000000000408002] (CPF_Const | CPF_Localized | CPF_NeedCtorLink)
};

// ScriptStruct OnlineSubsystemEOS.OnlinePlayerInterfaceEOS.LoginDelegates
// 0x0010
struct FLoginDelegates
{
	TArray<struct FScriptDelegate>                     Delegates;                                     // 0x0000 (0x0010) [0x0000000000400000] (CPF_NeedCtorLink)
};

/*
# ========================================================================================= #
#
# ========================================================================================= #
*/

#ifdef _MSC_VER
	#pragma pack(pop)
#endif
