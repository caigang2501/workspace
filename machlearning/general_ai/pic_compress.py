


class Solution:
    def gcdOfStrings(self, str1: str, str2: str) -> str:
        if str1+str2==str2+str1:
            if len(str1)<len(str2):
                str1,str2 = str2,str1
            i = 1
            while i < len(str2):
                print(i)
                if len(str2)%i==0 and len(str1)%(len(str2)//i)==0:
                    return str2[:len(str2)//i]
                i += 1
        return 'asdf'
    
s = Solution()
a = s.gcdOfStrings('aa','a')
print(a)

