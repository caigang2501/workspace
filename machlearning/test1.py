
class Solution:
    def maxChunksToSorted(self, arr) -> int:
        n = 0
        cut_line = -1
        def cutleft(arr):
            max_e = arr[0]
            max_e_temp = arr[0]
            index = 0
            for x in arr:
                if x > max_e:
                    max_e_temp = max(x,max_e_temp)
                else:
                    cut_line = index
                    max_e = max_e_temp
                index += 1
            return cut_line

        while cut_line != len(arr)-1:
            cut_line += cutleft(arr[cut_line+1:])+1
            n += 1
        return n
    
s = Solution()
l = [1,5,3,0,2,8,7,6,4]
print(s.maxChunksToSorted(l))