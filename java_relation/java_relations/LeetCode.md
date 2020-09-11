LeetCode

trie树，记住下面这张图

![无效的图片地址](https://pic.leetcode-cn.com/3463d9e7cb323911aa67cbd94910a34d88c9402a1ab41bbea10852cd0a74f2af-file_1562596867185)

​														`leet`在Trie树的显示

## Trie树插入和搜索操作

```java
package trietree.trie;

/**
 * @author codexwj
 * @CSDN https://blog.csdn.net/qq_31900497
 * @Github https://github.com/codexwj
 * @微信公众号 codexwj
 * @date 2020/5/2
 **/
public class Main {
    public static void main(String[] args) {

    }
}
class Trie{
    private boolean is_string = false;
    private Trie next[] = new Trie[26];

    public Trie(){}

    public void insert(String word){
        Trie root = this;
        char w[] = word.toCharArray();
        for (int i=0;i<w.length;i++){
            if(root.next[w[i]-'a'] == null){
                root.next[w[i]-'a'] = new Trie();//为空就新实例化一个Trie树。
            }
            root = root.next[w[i]-'a'];//移动根节点
        }
        root.is_string = true;
    }

    public boolean search(String word){
        Trie root = this;
        char w[] = word.toCharArray();
        for (int i=0;i<w.length;i++){
            if (root.next[w[i]-'a']== null) return false;
            root = root.next[w[i]-'a'];
        }
        return root.is_string;
    }

    public boolean startWith(String prefix){
        Trie root = this;
        char p[] = prefix.toCharArray();
        for (int i=0;i<p.length;++i){
            if(root.next[p[i]-'a']==null) return false;
            root = root.next[p[i]-'a'];
        }
        return true;
    }
}
```

### 20200503

**课程表||**：现在你总共有 n 门课需要选，记为 0 到 n-1。

在选修某些课程之前需要一些先修课程。 例如，想要学习课程 0 ，你需要先完成课程 1 ，我们用一个匹配来表示他们: [0,1]

给定课程总量以及它们的先决条件，返回你为了学完所有课程所安排的学习顺序。

可能会有多个正确的顺序，你只要返回一种就可以了。如果不可能完成所有课程，返回一个空数组。

```xml
输入: 4, [[1,0],[2,0],[3,1],[3,2]]
输出: [0,1,2,3] or [0,2,1,3]
解释: 总共有 4 门课程。要学习课程 3，你应该先完成课程 1 和课程 2。并且课程 1 和课程 2 都应该排在课程 0 之后。
     因此，一个正确的课程顺序是 [0,1,2,3] 。另一个正确的排序是 [0,2,1,3]
```

```java

```

![sum](https://pic.leetcode-cn.com/f2e99f870046c980322914283b457ca14634c2ef1c7ec949f02f6851c8dee9a4-image.png)

建立一个有向，无权图

由先修课程指向未修课程。

当形成循环时，表示没有学习的顺序。

深度优先搜索的流程是：在完成A的递归并移动到其他节点之前，我们将考虑源自A的所有路径，

```java
// 方法 2：邻接矩阵 + DFS
    // 用 HashSet 作为邻接矩阵，加速查找速度
    public int[] findOrder(int numCourses, int[][] prerequisites) {
        if (numCourses == 0) return new int[0];
        // HashSet 作为邻接矩阵
        HashSet<Integer>[] graph = new HashSet[numCourses];
        for (int i = 0; i < numCourses; i++) {
            graph[i] = new HashSet<>();
        }
        for (int[] p : prerequisites) {
            graph[p[1]].add(p[0]);
        }
        int[] mark = new int[numCourses]; // 标记数组
        Stack<Integer> stack = new Stack<>(); // 结果栈
        for (int i = 0; i < numCourses; i++) {
            if(!isCycle(graph, mark, i, stack)) return new int[0];
        }
        int[] res = new int[numCourses];
        for (int i = 0; i < numCourses; i++) {
            res[i] = stack.pop();
        }
        return res;
    }

    private boolean isCycle(HashSet<Integer>[] graph, int[] mark, int i, Stack<Integer> stack) {
        if (mark[i] == -1) return true;
        if (mark[i] == 1) return false;

        mark[i] = 1;
        for (int neighbor : graph[i]) {
            if (!isCycle(graph, mark, neighbor, stack)) return false;
        }
        mark[i] = -1;
        stack.push(i);
        return true;
    }
```

**数组中的第K个最大元素**

在未排序的数组中找到第 **k** 个最大的元素。请注意，你需要找的是数组排序后的第 k 个最大的元素，而不是第 k 个不同的元素。

**思路：**大顶堆

利用Comparable或Comparator自定义排序的方式，需要排序的数组中的数据得是引用类型，基本类型不能实现。

```java
Integer[] array = {3,4,5,2,1};
Arrays.sort(array, (s1, s2) -> {
  return s2.compareTo(s1);
});
输出：[5, 4, 3, 2, 1]
```

如上，可以实现排序，当定义int[] array是不能实现排序。

PriorityQueue的出队顺序和优先级有关，所以得实现Comparable接口。如果我们要放入的元素并没有实现`Comparable`接口怎么办？`PriorityQueue`允许我们提供一个`Comparator`对象来判断两个元素的顺序。我们以银行排队业务为例，实现一个`PriorityQueue`：

```java
public class Main {
    public static void main(String[] args) {
        Queue<User> q = new PriorityQueue<>(new UserComparator());
        // 添加3个元素到队列:
        q.offer(new User("Bob", "A1"));
        q.offer(new User("Alice", "A2"));
        q.offer(new User("Boss", "V1"));
        System.out.println(q.poll()); // Boss/V1
        System.out.println(q.poll()); // Bob/A1
        System.out.println(q.poll()); // Alice/A2
        System.out.println(q.poll()); // null,因为队列为空
    }
}

class UserComparator implements Comparator<User> {
    public int compare(User u1, User u2) {
        if (u1.number.charAt(0) == u2.number.charAt(0)) {
            // 如果两人的号都是A开头或者都是V开头,比较号的大小:
            return u1.number.compareTo(u2.number);
        }
        if (u1.number.charAt(0) == 'V') {
            // u1的号码是V开头,优先级高:
            return -1;
        } else {
            return 1;
        }
    }
}

class User {
    public final String name;
    public final String number;

    public User(String name, String number) {
        this.name = name;
        this.number = number;
    }

    public String toString() {
        return name + "/" + number;
    }
}
```

### 求得二叉搜索树的第k小的元素

给定一个二叉搜索树，编写一个函数 `kthSmallest` 来查找其中第 **k** 个最小的元素。

**须知**：二叉搜索树，又叫二叉排序树，二叉查找树。**特点是**：左子树的所有元素都小于等于根节点，右子树的所有节点都大于等于根节点。并且，**二叉搜索树的中序遍历是升序排列的**。

**自己的思路：**刚开始不知道二叉搜索树的性质；自己采用了优先队列的方式：

```java
    public int kthSmallest(TreeNode root, int k){
        PriorityQueue<Integer> pqueue = new PriorityQueue<Integer>((s1,s2)->s2-s1);
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        while (!queue.isEmpty()){
            TreeNode node = queue.poll();
            pqueue.add(node.val);
            if (pqueue.size()>k){
                pqueue.poll();
            }
            if (node.left != null){
                queue.add(node.left);
            }
            if (node.right!=null){
                queue.add(node.right);
            }
        }
        return pqueue.poll();
    }
```

但是效率并不好。

之后利用二叉搜索树的性质可以加快查找：利用栈来添加和移除元素。

```java
    public int kthSmallest2(TreeNode root, int k){
        LinkedList<TreeNode> stack = new LinkedList<>();
        while (true){
            while (root!=null){
                stack.add(root);
                root = root.left;
            }
            root  = stack.removeLast();
            if (--k==0) return root.val;
            root = root.right;
        }
    }
```

**复杂度分析**

- 时间复杂度：O(H+k)
- 空间复杂度：O(H+k)  H：树的高度。

### 二叉树的最近公共祖先

给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。

百度百科中最近公共祖先的定义为：“对于有根树 T 的两个结点 p、q，最近公共祖先表示为一个结点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。”

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/15/binarytree.png)

**示例**：

```xml
输入: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1
输出: 3
解释: 节点 5 和节点 1 的最近公共祖先是节点 3。
```

**解法：** 递归

假如根节点为x，两个需要找最近父节点的节点分别为q,p。

1. 当p和q分别为x的左子节点和右子节点时，x为根节点。
2. 假如x就是q、p中的一个，且另一个是x的左或右子节点。

```java
public boolean dfs(TreeNode root,TreeNode q,TreeNode p){
    if(root == null) return false;
    boolean lson = dfs(root.left,q,p);
    boolean rson = dfs(root.right,q,p);
    if((lson && rson) || ((q.val == root.val || p.val == root.val)&&(lson || rson))){
        ans = root;
    }
    return lson || rson || (p.val == root.val || q.val == root.val);
}
```

### 找到数组在[0,nums.length]缺失的一个数

```java
public int missingNumber(int[] nums){
    int missing = nums.length;
    for(int i=0;i<nums.length;i++){
        missing ^= i ^ nums[i];
    }
    retrun missing;
}
```

异或：两个相同的的值之间异或得到0，两个0之间再异或可以。

未缺失的数在 [0..n][0..n] 和数组中各出现一次，因此异或后得到 0。而缺失的数字只在 [0..n][0..n] 中出现了一次，在数组中没有出现，因此最终的异或结果即为这个缺失的数字。

babbad

### 和为K的子数组

```java
public int subArraySum(int[] nums,int k){
    if(nums == null || nums.length == 0){
        return 0;
    }
    int res = 0;
    for(int l=0;l<nums.length;l++){
        int sum=0;
        for(int r=l;r>=0;r--){
            sum += nums[r];
            if(sum == k){
                res ++; 
            }
        }
    }
    return res;
}
```

****

**前缀和**

```java
public int subArraySum(int[] nums,int k){
    int pre=0,count = 0;
    HashMap<Integer,Integer> map = new HashMap<>();
    map.put(0,1);
    for(int i=0;i<nums.length;i++){
        pre += nums[i];
        if(map.containsKey(pre-k)){
            count += map.get(pre-k);
        }
        map.put(pre,map.getOrDefault(pre,0)+1);
    }
    return count;
}
```

### 整数反转

```java
public reverse(int x){
    int rev=0;
    while(x!=0){
        int pop = x % 10;
        x /= 10;
        if(x > Integer.MAX_VALUE / 10 || (x = Integer.MAX_VALUE / 10&&pop == 7)) return 0;
        if(x < Integer.MIN_VALUE / 10 || (x = Integer.MIN_VALUE /10  && pop == -8))
            return 0;
        rev = rev * 10 + pop;
    }
    return rev;
}
```

### 默写课程表||

```java
public int[] findOrder(int numberCourse,int[][] prerequisites){
    Stack<Integer> stack = new Stack<>();
    HashSet[] graph = new HashSet[numberCourse];
    for(int i=0;i<numberCourse;i++){
        graph[i] = new HashSet<>();
    }
    for(int[] p : prerequisites){
        graph[p[1]].add(p[0]);
    }
    int[] marked = new int[numberCourse];
    for(int i=0;i<numberCourse;i++){
        if(!dfs(graph,marked,i,stack)) return new int[0];
    }
    int[] res = new int[numberCourse];
    int k=0;
    while(!stack.isEmpty()){
        res[k++] = stack.pop();
    }
    return res;
}

public boolean dfs(hashSet<Integer>[] graph,int[] marked,int i, Stack<Integer> stack){
    if(marked[i] == -1) return true;
    if(marked[i] == 1) return false;
    marked[i] = 1;
    for(int neighbor:graph[i]){
        if(!dfs(graph,marked,neighbor,stack)) return false;
    }
    marked[i]=-1;
    stack.push(i);
    return true;
}
```

### 乘积最大子数组

**方法**：动态规划

由于乘法的特性，可能正数称以正数，负数乘以负数。此时的动态规划的转移方程为：

![image-20200518093831748](C:\Users\codexwj\AppData\Roaming\Typora\typora-user-images\image-20200518093831748.png)

**code**

```java
class Solution{
    private int min(int x, int y){return x < y?x:y;}
    private int max(int x, int y){return x > y?x:y;}
    
    public int maxProduct(int[] nums){
        int[] maxF = new int[nums.length];
        int[] minF = new int[nums.length];
        
        maxF[0] = nums[0];
        minF[0] = nums[0];
        int ans = nums[0];
        
        for(int i=1;i<nums.length;i++){
            maxF[i] = max(maxF(i-1) * nums[i],max(minF(i-1)*nums[i],nums[i]));
            minF[i] = min(maxF(i-1) * nums[i],min(minF(i-1)*nums[i],nums[i]));
            ans = max(ans,maxF[i]);
        }
        return ans;
    }
}
```

```java
class Solution{
    public int maxProduct(int[] nums){
        int max = Integer.MIN_VALUE,imax = 1, imin = 1;
        for(int i=0;i<nums.length;i++){
            if(nums[i] < 0){
                int tmp = imax;
                imax = imin;
                imin = tmp;
            }
            imax = Math.max(imax*nums[i],nums[i]);
            imax = Math.min(imin*nums[i],nums[i]);
            max = Math.max(max,imax);
        }
        return max;
    }
}
```

### 删除一个字符，判断该字符串能不能为回文串

双指针，先通过指针去除字符串首尾相等的字符，在进行判断是否是low == high-1或者low+1 == high；

```java
class Solution{
    public boolean validPalindrome(String s){
        int low = 0,high = s.length() - 1;
        while(low < high){
            char c1 = s.charAt(low), c2 = s.charAt(high);
            if(c1 == c2){
                low++;
                high--;
            }else{
                boolean flag1 = true, flag2 = true;
                for(int i=low,j=high-1;i<j;i++,j--){
                    char c3 = s.charAt(i),c4 = s.charAt(j);
                    if(c3 != c4){
                        flag1 = false;
                        break;
                    }
                }
                for(int i=low+1,j=high;i<j;i++,j--){
                    char c5 = s.charAt(i),c6 = s.charAt(j);
                    if(c5 != c6){
                        flag2 = false;
                        break;
                    }
                }
                return flag1 || flag2;
            }
            
        }
        return true;
    }
}
```

### 盛最多水的容器

双指针：思路：当移动低档板的时候，容器可能会变大，移动高档版容量一定减少。

```java
class Solution{
    public int maxArea(int[] heigth){
        int low = 0,high = heigth.length - 1;
        int maxA = 0;
        while(low < high){
            maxA = Math.max(maxA,Math.max(heigth[low],heigth[high])*(high - low));
            if(height[high]>heigth[low]){
                low++;
            }else{
                high--;
            }
        }
        return maxA;
    }
}
```

### 罗马数字转整数

思路：判断前一个数字是否比当前值小还是大，如果小就相减、大就相加。

```java
class Solution{
    public int romanToInt(String s){
        int sum = 0;
        int preNum = getValue(s.charAt(0));
        for(int i=1;i<s.length();i++){
            int num = getValue(s.charAt(i));
            if(preNum < num){
                sum -= preNum;
            }else{
                sum += preNum;
            }
            preNum = num;
        }
        sum += preNum;
        return sum;
    }
    
    public int getValue(char ch){
        switch(ch){
            case 'I' : return 1;
            case 'V': return 5;
            case 'X': return 10;
            case 'L': return 50;
            case 'C': return 100;
            case 'D': return 500;
            case 'M': return 1000;
            default: return 0; 
        }
    }
}
```

### 最长公共前缀

```java
class Solution{
    public String longestCommonPrefix(String[] strs){
        if(strs.length == 0||strs == null) return "";
        for(int i=0;i<strs[0].length();i++){
            char ch = strs[0].charAt(i);
            for(int j=1;j<strs.length;j++){
                if(i==strs[j].length() || ch != strs[j].charAt(i)){
                    return strs[0].substring(0,i);
                }
            }
        }
        return strs[0];
    }
}
```

**分治**

```java
private String longestCommonPrefix(String[] strs,int l, int r){
    if(l == r){
        return strs[l];
    }
    while(l < r){
        int mid = (l + r)/2;
        String lcpLeft = longestCommonPrefix(strs,l,mid);
        String lcpRight = longestCommonPrefix(strs,mid+1,r);
        return commonPrefix(lcpLeft,lcpRight);
    }
}
String commonPrefix(String left,String right){
    int min = Math.min(left.length(),right.length());
    for(int i=0;i<min;i++){
        if(left.charAt(i)!=right.charAt(i)){
            return left.substring(0,i);
        }
    }
    return left.substring(0,min);
}
```

### 有效的括号

**思路**：利用map进行配对，key为左括号，value为右括号。在遍历括号时，遇到左括号则将其压栈，不包含时遍历值与value值进行比较。

```java
class Solution{
    private static final Map<Character,Character> map = new HashMap<>(){
        {
            put('{','}');put('[',']');put('(',')');put('?','?');
        }
    };
    public boolean isValid(String s){
        if(s.length()>0&&!map.containsKey(s.charAt(0))) return false;
        Stack<Character> stack = new Stack<>();
        for(Character c : s.toCharArray()){
            if(map.containsKey(c)) stack.push(c);
            else if(stack.isEmpty() || map.get(stack.pop()) != c){
                return false;
            }
        }
        return stack.isEmpty();
    }
}
```

### 电话号码的字母组合

**步骤**

1.建立映射

2.取出第一个数字，同时取出数字对应的字符串

3.对取出的字符串进行遍历，把遍历的字符串添加到暂存的字符串中，同时递归调用函数，参数改为下一个数字。



### 生成括号

**方法**：深度优先+减枝

1.需要满足左右括号大于0

2.首先产生左分支

3.产生右分支时，需要满足可使用的左括号大于右括号

```java
class Solution{
    public List<String> generateParenthesis(int n){
        List<String> res = new ArrayList<>();
        
        if(n == 0){
            return res;
        }
        dfs("",n,n,res);
        return res;
    }
    public void dfs(String curStr,int left,int right,List<String> res){
        if(left == 0 && right == 0){
            res.add(curStr);
            return;
        }
        
        if(left > right){
            return;
        }
        if(left >0){
            dfs(curStr+"(",left-1,right,res);
        }
        if(right >0){
            dfs(curStr+")",left,right-1,res);
        }
    }
}
```

### 扫雷

递归结束条件：遇到有雷和到达边际；

```java
class Solution{
    public char[][] updateBoard(char[][] board,int[] click){
        if(board[click[0]][click[1]] == 'M'){
            board[click[0][click[1]]] = 'X';
            return;
        }
        return click(board,click[0],click[1]);
    }
    private char[][] click(char[][] board,int x,int y){
        int num = getNum(board, x,y);
        if(num == 0){
            board[x][y] = 'B';
        }else{
            board[x][y] = Character.forDigit(num,10);
            return board;
        }
        for(int i=-1;i<=1;i++){
            for(int j=-1;j<=1;j++){
                if(x + i >= 0 && x + i < board.length&&y + j >=0&&y+j<board[0].length&&board[x+i][y+j]=='E'){
                    board = click(board,x+i,y+j);
                }
            }
        }
        return board;
    }
    private int getNum(char[][] board,int x,int y){
        int num = 0;
        for(int i=-1;i<=1;i++){
            for(int j=-1;j<=1;j++){
                if(x + i >= 0&&y + j >=0&&x+i<board.length&&y+j<board[0].length&&board[x+i][y+j]=='M'){
                    num ++;
                }
            }
        }
        return num;
    }
}
```

### 飞地的数量

```java
class Solution{
    private int sum = 0;
    private int number = 0;
    private boolean flag = false;
    private int[] dx = {-1,1,0,0};
    private int[] dy = {0,0,-1,1};
    public int numEnclaves(int[][] A){
        int R = A.length;
        int C = A[0].length;
        int ans;
        boolean[][] visited = new boolean[R][C];
        for(int i=0;i<R;i++){
            for(int j=0;j<C;j++){
                if(A[i][j] == 1){
                    number = 0;
                   dfs(A,i,j); 
                   if(!flag){
                       sum+=number;
                   }
                    flag = false;
                }
            }
        }
        return sum;
    }
    
    public void dfs(int[][] A,int x,int y){
        if(x < 0 || x >= A.length || y<0 || y >= A[0].length){
            flag = true;
            return;
        }
        if(A[x][y] != 1){
            return;
        }
        number++;
        A[x][y]=0;
		for(int n=0;n<4;n++){
            dfs(A,x + dx[n], y+dy[n]);
        }
    }
}
```

### 把所有和边际相连的陆地淹没，剩下的就是飞路了

```java
class Solution{
    public int numEnclaves(int[][] A){
        int r = A.length;
        int c = A[0].length;
        for(int i=0;i<r;i++){
            dfs(A,i,0);
            dfs(A,i,c-1);
        }
        for(int i=0;i<c;i++){
            dfs(A,0,i);
            dfs(A,r-1,i);
        }
        int count = 0;
        for(int i=0;i<r;i++){
            for(int j=0;j<c;j++){
                if(A[i][j]==1){
                    count++;
                }
            }
        }
        return count;
    }
    private void dfs(int[][] a,int x,int y){
        if(a[x][y]==0) return;
        a[x][y] = 0;
        if(x > 0) dfs(a,x-1,y);
        if(y > 0) dfs(a,x,y-1);
        if( x < a.length - 1) dfs(a,x+1,y);
        if(y < a[0].length - 1) dfs(a,x,y-1);
    }
}

///
class Solution {
    int row, col;
    int[][] A;
    public int numEnclaves(int[][] A) {
        if(A == null || A.length == 0) return 0;
        this.A = A;
        this.row = A.length;
        this.col = A[0].length;

        // 淹没所有和边界相接的陆地
        for (int i = 0; i < row; i++) {
            dfs(i, 0);
            dfs(i, col - 1);
        }
        for (int i = 0; i < col; i++) {
            dfs(0, i);
            dfs(row - 1, i);
        }
        // 统计剩下的飞陆
        int count = 0;
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                if(A[i][j] == 1) count++;
            }
        }
        return count;
    }

    /**
     * 把此大陆淹没，即把 1 变成 0
     * @param r,c DFS 起点
     */
    public void dfs(int r, int c){
        if(A[r][c] == 0) return;

        A[r][c] = 0;
        if(r > 0       ) { dfs(r - 1, c);       }
        if(c > 0       ) { dfs(r,     c - 1);   }
        if(r < row - 1 ) { dfs(r + 1, c);       }
        if(c < col - 1 ) { dfs(r,     c + 1);   }
    }
}
```

### 利用前序遍历和中序遍历构造一个二叉树

**题解**：根据前序遍历和中序遍历的特点，前序遍历头节点是子树的根节点，而中序遍历的头节点在子树的"`中间`"。所以可以根据前序遍历的头结点把二叉树分成两部分。

```java
class Solution{
    public TreeNode buildTree(int[] preOrder, int[] inOrder){
        return helper(preOrder,inOrder,0,0,inOrder.length - 1);
    }
    private TreeNode helper(int[] preOrder, int[] inOrder, int preStart,int inStart,int inEnd){
        if(inStart > inEnd) return null;
        int currentVal = preOrder[preStart];
        TreeNode current = new TreeNode(currentVal);
        int index = 0;
        for(int i = inStart;i<=inEnd;i++){
            if(inOrder[i] == currentVal){
                index = i;
            }
        }
        TreeNode l = helper(preOrder,inOrder,preStart + 1,inStart, index - 1);
        TreeNode r = helper(preOrder,inOrder,preStart + (index-inStart) + 1, index + 1, inEnd);
        current.left = l;
        current.right = r;
        return current;
    }
}
```

#### 利用后序和中序遍历生成一棵二叉树

```java
class Solution{
    public TreeNode buildTree(int[] inorder,int[] postorder){
        return helper(inorder, postorder,postorder.length - 1,0,inorder.length - 1);
    }
    private TreeNode helper(int[] inorder, int[] postorder,int postEnd, int inStart, int inEnd){
        if(inStart > inEnd){
            return null;
        }
        int currentVal = postorder[postEnd];
        TreeNode current = new TreeNode(currentVal);
        
        int index = 0;
        for(int i=inStart;i <= inEnd;i++){
            if(inorder[i] == currentVal){
                index = i;
            }
        }
        TreeNode l = helper(inorder, postorder,postEnd - (inEnd - index) - 1,inStart,index -1);
        TreeNode r = helper(inorder,postorder,postEnd-1,index + 1, inEnd);
        current.left = l;
        current.right = r;
        return current;
    }
}
```

### 被包围的区域，和边界相连的区域是一种特殊的区域，需要单独处理。

#### 递归方式

```java
class Solution{
    public void solve(char[][] board){
        if(board == null || board.length == 0) return;
        int row = board.length;
        int col = board[0].length;
        for(int i=0;i<row;i++){
            for(int j=0;j<col;j++){
                boolean edge = false;
                if(i==0||j==0||i==row-1||j==col-1){
                    edge = true;
                }
                if(edge&&board[i][j] == 'O'){
                    dfs(board,i,j);
                }
            }
        }
        for(int i=0;i<row;i++){
            for(int j=0;j<col;j++){
                if(board[i][j] == 'O'){
                    board[i][j] = 'X';
                }
                if(board[i][j] == '#'){
                    board[i][j] = 'O';
                }
            }
        }
    }
    public void dfs(char[][] board, int r,int c){
        if(c<0||r<0||r>board.length-1||c>board[0].length-1||board[r][c] =='#'||board[r][c]=='X'){
            return;
        }
        board[r][c] = '#';
        dfs(board,r-1,c);
        dfs(board,r+1,c);
        dfs(board,r,c-1);
        dfs(board,r,c+1);
    }
}
```

#### 非递归的方式

```java
class Solution{
    public class Pos{
        int i;
        int j;
        public Pos(int i, int j){
            this.i = i;
            this.j = j;
        }
    }
    public void solve(char[][] board){
        if(board == null || board.length == 0) return;
        int row = board.length;
        int col = board[0].length;
        for(int i=0;i<row;i++){
            for(int j=0;j<col;j++){
                boolean edge = i == 0||i==row -1||j==0||j==col-1;
                if(edge&&board[i][j] == 'O'){
                    dfs(board,i,j);
                }
            }
        }
    }
    public void dfs(char[][] board,int i, int j){
        Stack<Pos> stack = new Stack<>();
        board[i][j] = '#';
        stack.push(new Pos(i,j));
        while(!stack.isEmpty()){
            Pos current = stack.peek();
            //上
            if(current.i-1 >= 0 && board[current.i-1][current.j]=='O'){
                stack.push(new Pos(current.i-1,current.j));
                board[current.i-1][current.j] = '#';
                continue;
            }
            //下
            if(current.i+1 < board.length && board[current.i+1][current.j]=='O'){
                stack.push(new Pos(current.i+1,current.j));
                board[current.i+1][current.j] = '#';
                continue;
            }
            //左
            if(current.j-1 >= 0 && board[current.i][current.j-1]=='O'){
                stack.push(new Pos(current.i,current.j-1));
                board[current.i][current.j-1] = '#';
                continue;
            }
            //右
            if(current.j+1 < board[0].length && board[current.i][current.j+1]=='O'){
                stack.push(new Pos(current.i,current.j+1));
                board[current.i][current.j+1] = '#';
                continue;
            }
            //如果上下都搜不到，本次搜索结束，弹出。
            stack.pop();
        }
    }
}
```

在dfs中只要我们满足一个方向就会沿着这个方向一直搜索。所有会用到continue。而bfs需要把上下左右四个方向都入队，所以搜索的时候不能用continue。

### 3数之和

**题目**： 给定一个数组找出3个数之和为零的元素。

**思路** ：排序 + 双指针；

- 首先对数组进行排序
- 排序之后3个指针中第一个指针k进行固定，对于另外两个指针i，j可以在(k,len(n))之间取值，通过双指针交替向中间移动，记录对于每个固定的指针k，找出满足，nums[k]+nums[i]+nums[j]=0的i,j组合。
  - 当nums[k]>0时，每个指针指向的元素都是大于零的，因此进行返回。
  - 当k>0时，对满足nums[k]=nums[k-1]时跳过此元素；
  - 对于i,j分别设在数组的（k,len(nums)）两端，当i<j时循环计算nums[k]+nums[i]+nums[j]==0。
    - 当s<0时，此时需要移动左指针，并需要跳过所有和nums[i]相同的元素。
    - 当s>0时，此时需要移动右指针，并需要跳过所有和nums[j]相同的元素。
    - 当s==0时，进行记录，执行i+1，j-1，并跳过重复的元素。

```java
class Solution{
    public List<List<Integer>> threeSum(int[] nums){
        List<List<Integer>> res = new LinkedList<>();
        if(nums == null || nums.length < 3) return res;
        Arrays.sort(nums);
        for(int k=0;k<nums.length-2;k++){
            if(nums[k] > 0) break;
            if(k > 0 && nums[k]==nums[k-1]) continue;
            int i=k+1, j=nums.length - 1;
            while(i<j){
                int sum = nums[k] + nums[i] + nums[j];
                if(sum < 0){
                    while(i<j&&nums[i]==nums[++i]);
                }
                else if(sum > 0){
                    while(i<j&&nums[j]==nums[--j]);
                }
                else {
                    res.add(new LinkedList<Integer>(Arrays.asList(nums[k],nums[i],nums[j])));
                    while(i<j&&nums[i]==nums[++i]);
                    while(i<j&&nums[j]==nums[--j]);
                }
            }
        }
        return res;
    }
}
```

### 电话号码的字母组合

排序组合：回溯算法；

**思路**：首先需要对数字和字母建立映射，建立映射之后，需要取出每个数字，并且通过这个数字找到映射的字符串，之后对这个字符串进行遍历，以每个字符开始进行递归回溯。

- 跳出递归的条件：当数字的长度等于零的时候结束递归，添加结果。
- 此时进行回递，回到for循环，遍历下一个字符。

```java
class Solution{
    List<String> output = new ArrayList<>();
    HashMap<String,String> map = new HashMap<>(){{
    put("2", "abc");
    put("3", "def");
    put("4", "ghi");
    put("5", "jkl");
    put("6", "mno");
    put("7", "pqrs");
    put("8", "tuv");
    put("9", "wxyz");
  }};

    public List<String> letterCombinations(String digits){
        if(digits.length() !=0){
            backtrack("",digits);
        }
        return output;
    }
    public void backtrack(String combination, String next_digits){
        if(next_digits.length() == 0){
            output.add(combination);
            return;
        } else{
            String digit = next_digits.substring(0,1);
            String letters = map.get(digit);
            for(int i=0;i<letters.length();i++){
                String letter = letters.substring(i,i+1);
                backtrack(combination + letter, next_digits.substring(1));
            }
        }
    }
}
```

### 行星碰撞

**思路** ：假设栈中顶部元素为top，一颗新的小行星new进来。如果new向右移动(new>0)，或者top向左移动（top < 0），则

- 否则，如果abs(new) < abs(top)，则新小行星new将爆炸；如果abs（new）== abs(top)，则两个小行星将爆炸；如果abs(new) > abs(top)，则top小行星爆炸。
- 
- 1，栈顶元素大于零，新元素小于零，且栈顶元素的绝对值小于新元素的绝对值，出栈。
- 2，当两元素的绝对值相等时，相互抵消。
- 循环1和2：
- 当栈为空时，栈顶元素<0，新元素大于零，入栈。

```java
class Solution{
    public int[] asteroidCollision(int[] asteroids){
        ArrayDeque<Integer> stack = new ArrayDeque<>();
        loop:
        for(int as : asteroids){
            while(as < 0 && !stack.isEmpty()&&stack.peek() > 0&& (stack.peek()+as)<=0){
                int top = stack.pop();
                if(top + as == 0){
                    continue loop;
                }
            }
            if(stack.isEmpty() || stack.peek() < 0 || as > 0){//attention
                stack.push(as);
            }
        }
        int size = stack.size();
        int[] res = new int[size];
        for(int i=size-1;i>=0;i--){
            res[i] = stack.pop();
        }
        return res;
    }
}
```

两者有什么不同：

```java
    public int[] asteroidCollision(int[] asteroids) {
        ArrayDeque<Integer> stack = new ArrayDeque<>();
        loop:
        for(int out: asteroids){
            while(out<0 && !stack.isEmpty() && stack.peek()>0 && stack.peek()+out<=0){
                int top = stack.pop();
                if(top+out==0) continue loop;
            }
            if(stack.isEmpty() || stack.peek()<0 || out>0) stack.push(out); 
        }
        int size = stack.size();
        int[] ans = new int[size];
        for(int i = size-1; i>=0; i--){
            ans[i] = stack.pop();
        }
        return ans;
    }
```

#### 求pow(x,n)

思路：利用递归

```java
class Solution{
    public double helper(double x, int n){
        if(n == 0){
            return 1.0;
        }
        double y = helper(x,n/2);
        return n%2 == 0?y*y?y*y*x;
    }
	public double myPow(double x,int n){
        long N = n;
        return N>=0?helper(x,N):1.0/helper(x,-N);
        
    }
}
```

#### 搜索旋转排序数组

思路：二分查找，部分有序也是可以的。

```java
class Solution{
    public int search(int[] nums, int target){
        int n = nums.length;
        if(n == 0) return -1;
        if(n == 1) return nums[0] == target ? 0:-1;
        int l = 0, r = n-1;
        while(l <= r){
            int mid = (l + r) / 2;
            if(nums[mid] == target) return mid;
            if(nums[0]<=nums[mid]){
                if(nums[0]<=targrt && target < nums[mid]){
                    r = mid - 1;
                }else{
                    l = mid + 1;
                }
            }else{
                if(nums[mid]<target&&target <= nums[n-1]){
                    l = mid + 1;
                }else{
                    r = mid - 1;
                }
            }
        }
        return -1;
    }
}
```

****

#### 有序矩阵中第k小的元素

**题目**：给定一个n×n的矩阵，其中每行和每列元素均升序排列，找到矩阵第k个元素，它是排序后的第k个小元素。

**方法一**：优先队列

```java
public int kthSmallest(int[][] matrix, int k){
    PriorityQueue<Integer> MaxPQ = new PriorityQueue<>(Collections.reverseOrder());
    for(int[] row : matrix){
        for(int num : row){
            if(MaxPQ.size() == k && num > MaxPQ.peek()){
                break;
            }
            MaxPQ.add(num);
            if(MaxPQ.size()>k){
                MaxPQ.remove();
            }
        }
    }
    return MaxPQ.remove();
}
```

**方法二**：二分法

解法：

![fig3](https://assets.leetcode-cn.com/solution-static/378/378_fig3.png)

步骤：

可以这样描述走法：

初始位置在$ matrix[n - 1][0]$（即左下角）；

设当前位置为$matrix[i][j]。若 $matrix[i][j] \leq mid$，则将当前所在列的不大于mid 的数的数量（即 i + 1）累加到答案中，并向右移动，否则向上移动；

不断移动直到走出格子为止。

我们发现这样的走法时间复杂度为 O(n)，即我们可以线性计算对于任意一个 mid，矩阵中有多少数不大于它。这满足了二分查找的性质。

不妨假设答案为 x，那么可以知道$ l\leq x\leq r$，这样就确定了二分查找的上下界。

每次对于「猜测」的答案 mid，计算矩阵中有多少数不大于 mid ：

如果数量不少于 k，那么说明最终答案 x 不大于 mid；
如果数量少于 k，那么说明最终答案 x大于 mid。
这样我们就可以计算出最终的结果 x 了。

```java
public int kthSmallest(int[][] matrix, int k){
    int n = matrix.length-1;
    int left = matrix[0][0],right = maxtri[n][n];
    while(left < right){
        int mid = left + (right-left)/2;
        int count = countNotMoreThanMid(matrix,mid,n);
        if(count<k){
            left = mid + 1;
        }else{
            right = mid;
        }
    }
    return left;
}
public int countNotMoreThanMid(int[][] matrix, int mid, int n){
    int count = 0;
    int x = 0, y=n;
    while(x <= n && y >= 0){ //之前少写了一个==号
      if(matrix[y][x] <= mid){
        count += y + 1;
        x++;
      }else{
          y--;
      }    
    }
	return count;
}
```

#### 排序复习



```java
public void bubbleSort(int[] arr){
    boolean flag = false;
    for(int i=0;i<arr.length;i++){
        for(int j=0;j<arr.length-1-i;j++){
            if(arr[j+1]<arr[j]){
                flag  = true;
                int temp = arr[j];
                arr[j] = arr[j+1];
                arr[j+1] = temp;
            }
        }
        if(!flag){
            break;
        }else{
            flag = true;
        }
    }
}


public void selectSort(int[] arr){
    for(int i=0;i<arr.length-1;i++){
        int min = arr[i];
        int minIndex = i;
        for(int j=i+1;j<arr.length;j++){
            if(arr[j]<min){
                min = arr[j];
                minIndex = j;
            }
        }
        if(minIndex != i){
            arr[minIndex] = arr[i];
            arr[i] = min;
        }
    }
}

//插入排序
public void InsertSort(int[] arr){
    int insertVal;
    int insertIndex;
    for(int i=1;i<arr.length;i++){
        insertVal = arr[i];
        insertIndex = i - 1;
        while(insertIndex >= 0 && insertVal < arr[insertIndex]){
            arr[insertIndex++] = arr[insertIndex];
            insertIndex--;
        }
        if(insertIndex +1 != i){
            arr[insertIndex+1] = insertVal;
        }
    }
}

//quickSort

```

### 检查平衡性

判断一棵树是否平衡：对于任何一个节点，其两棵子树的高度不超过1。

**示例**1

```java
给定二叉树 [3,9,20,null,null,15,7]
    3
   / \
  9  20
    /  \
   15   7
返回 true 。
```

**思路**：递归遍历找到高度。

**难点**：获取树的高度，递归遍历左右节点。

```java
class Solution{
    public boolean isBalanced(TreeNode root){
        if (root == null) return true;
        if(Math.abs(getDepth(root.left) - getDepth(root.right)) > 1) return false;
        return isBalanced(root.left)&&isBalanced(root.right);
    }
    
    private int getDepth(TreeNode root){
        if(root == null) return 0;
        return Math.max(getDepth(root.left),getDepth(root.right)) + 1;
    }
}

public class TreeNode{
    int val;
    TreeNode left;
    TreeNode right;
    TreeNode(int x){
        val = x;
    }
}
```

**复杂度**

1.时间复杂度：O(n)

2.空间复杂度：

### 路径总和

****

**递归的重要思想是**：1.找到最简单的子问题求解。2其他问题不考虑内在细节，只考虑整体逻辑。

**最简单的子问题，递归停止的条件**

```java
if(root == null){
    return 0;
}
```

**根据题目要求**

考虑三部分

- 以当前节点作为头节点的路径数量
- 以当前节点的左子树作为头节点的路径数量
- 以当前节点的右子树作为头结点的路径数量

**难点**：怎样去求以当前节点作为头节点的路径数量？

答案：每到一个节点让sum-root.val，并判断sum是否为0，如果为零的话，则找到满足条件的一条路径。

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    public int pathSum(TreeNode root, int sum) {
        if(root == null){
            return 0;
        }
        int result = countPath(root,sum);
        int a = pathSum(root.left,sum);
        int b = pathSum(root.right,sum);
        return result+a+b;

    }
    public int countPath(TreeNode root,int sum){
        if(root == null){
            return 0;
        }
        sum = sum - root.val;
        int result = sum == 0 ? 1:0;
        return result + countPath(root.left,sum) + countPath(root.right,sum);
    }
}
```

### 二叉搜索树迭代器

实现一个二叉搜索树迭代器，你将使用二叉搜索树的根节点初始化迭代器。调用`next()`将返回二叉搜索树中的下一个最小的数。

**知识点**：来看看迭代器

```java
new_iterator = BSTIterator(root);
while(new_iterator.hasNext())
    process(new_iterator.next());
```

**重点**：二叉搜索树的一个重要特性是二叉树搜索的中序遍历是升序遍历；因此，中序遍历是该解决方案的核心。

**方法一**：用一个空数组来存放二叉搜索树的中序序列。



```java
class BSTIterator {
	ArrayList<Integer> nodeSorted;
    int index;
    public BSTIterator(TreeNode root) {
		this.nodeSorted = new ArrayList<>();
        
        this.index = -1;
        
        this._inorder(root);
    }
  
    private void _inorder(TreeNode root){
        if(root == null) return;
        
        this._inorder(root.left);
        this.nodeSorted.add(root.val);
        this._inorder(root.right);
    }
    
    /** @return the next smallest number */
    public int next() {
		return nodeSorted.get(++this.index);
    }
    
    /** @return whether we have a next smallest number */
    public boolean hasNext() {
		return this.index + 1 < this.nodeSorted.size();
    }
}
```

```java
class BSTIterator{
    Stack<TreeNode> stack;
    
    public BSTIterator(TreeNode root){
        this.stack = new Stack<TreeNode>();
        this._leftmostInorder(root);
    }
    
    private void _leftmostInorder(TreeNode root){
        while(root!=null){
            this.stack.push(root);
            root = root .left;
        }
    }
    public int next(){
        TreeNode topmostNode = this.stack.pop();
        if(topmostNode.right != null){
            this._leftmostInorder(topmostNode.right);
        }
        return topmostNode.val;
    }
    
    public boolean hasNext(){
        return this.stack.size() > 0;
    }
}
```

### 二叉树的右视图

给定一颗二叉树，想象自己站在他的右侧，按照从顶部到底部的顺序，返回从右侧能看到的节点值。

**示例**：

```java
输入: [1,2,3,null,5,null,4]
输出: [1, 3, 4]
解释:

   1            <---
 /   \
2     3         <---
 \     \
  5     4       <---
```

**难点**：不知道树长什么样子。

![fig1](https://assets.leetcode-cn.com/solution-static/199_fig1.png)

**思路**：进行深度优先搜索，对树进行优先搜索时，先访问右子树。

这个题用树的深度进行判断是否需要向`map`中添加节点，利用了一个`map`来保持当前节点的高度，当高度小于等于`map`中储存的高度时，不进行存储当前节点。

```java
class Solution{
    public List<Integer> rightSideView(TreeNode root){
        Map<Integer,Integer> rightmostValueAtDepth = new HashMap<Integer,Integer>();
        int max_depth = -1;
        
        Stack<TreeNode> nodeStack = new Stack<TreeNode>();
        Stack<Integer> depthStack = new Stack<Integer>();
        nodeStack.push(root);
        depthStack.push(0);
        
        while(!nodeStack.isEmpty()){
            TreeNode node = nodeStack.pop();
            int depth = depthStack.pop();
            
            if(node != null){
                max_depth = Math.max(max_depth,depth);
                if(!rightmostValueAtDepth.containsKey(depth)){
                    rightmostValueAtDepth.put(depth, node.val);
                }
                nodeStack.push(node.left);
                nodeStack.push(node.right);
                depthStack.push(depth + 1);
                depthStack.push(depth + 1);
            }
        }
        List<Integer> rightView = new ArrayList<Integer>();
        for(int depth = 0;depth <= max_depth;depth++){
            rightView.add(rightmostValueAtDepth.get(depth));
        }
    }
}
```

**方法二**：通过层次遍历，对每一行的最后一个节点进行保存。

```java
class Solution{
    public List<Integer> rightSideView(TreeNode root){
        if(root == null){
            return new ArrayList<>();
        }
        Queue<TreeNode> queue = new LinkedList<>();
        List<Integer> ret = new ArrayList<>();
        queue.add(root);
        while(!queue.isEmpty()){
            int size = queue.size();
            for(int i = 0;i<size;i++){
                TreeNode node = queue.poll();
                if(i == size - 1){
                    ret.add(node.val);
                }
                if(node.left != null){
                    queue.add(node.left);
                }
                if(node.right != null){
                    queue.add(node.right);
                }
            }
        }
        return ret;
    }
}
```

**思路**：优先右子树的深度优先遍历，`deepest`表示目前存的最大深度

```java
class Solution{
    List<Integer> ans = new ArrayList<>();
    int deepest = 0;
    
    public List<Integer> rightSideView(TreeNode root){
        helper(root,0);
        return ans;
    }
    
    private void helper(TreeNode root, int now){
        if(root == null) return;
        if(now == deepest){
            ans.add(root.val);
            deepest++;
        }
        helper(root.right,now+1);
        helper(root.left, now+1);
    }
}
```



### 顺时针打印矩阵

**方法**：模拟打印矩阵的路径，初始位置是矩阵的左上角，初始方向是向右，当路径超过界限或者进入之间访问的位置时，则顺时针旋转，进入下一个方向。

**难点**：怎样定义方向，怎样判断和更新方向。

```java
class Solution{
    public int[] spiralOrder(int[][] matrix){
        if(matrix == null){
            return new int[0];
        }
        int rows = matrix.length, columns = matrix.length;
        boolean[][] visited = new boolean[rows][columns];
        int total = rows * columns;
        int[] order = new int[total];
        int row = 0, column = 0;
        int[][] directions = {{0,1},{1,0},{0,-1},{-1,0}};
        int directionIndex = 0;
        for(int i=0;i<total;i++){
            order[i] = matrix[row][column];
            visited[row][column] = true;
            int nextRow = row + dirctions[dirctionIndex][0], nextColumn = column + dirctions[dirctionIndex][1];
            if(nextRow < 0 || nextRow >= rows || nextColumn < 0 || nextColumn >= columns || visited[nextRow][nextColumn]){
                directionIndex = (directionIndex + 1) % 4;
            }
            row += directions[directionIndex][0];
            column += dirctions[directionIndex][1];
        }
        return order;
    }
}
```

### 二叉树的序列化和反序列化

**二叉树序列化的本质就是对其值和结构进行编码**

对于DFS搜索根据根节点、左节点和右节点之间的相对顺序，可进一步将DFS策略分为：

- 先序遍历
- 中序遍历
- 后序遍历

**对于这个题我的难点**，怎样控制输出空节点时，再怎样进行递归返回。

本题用了字符串连接的方式连接节点和`None`。

```java
public class Codec{
    public String reseriable(TreeNode root, String str){
        if(root == null){
            str += "None";
        }else{
            str += str.valueOf(root.val) + ",";
            str = reseriable(root.left, str);
            str = reseriable(root.right,str);
        }
        return str;
    }
    
    public String serialize(TreeNode root){
        return reseriable(root,"");
    }
    
    public TreeNode rdeseriable(List<String> l){
        if(l.get(0).equals("None")){
            l.remove(0);
            return null;
        }
        
        TreeNode root = new TreeNode(Integer.valueOf(l.get(0)));
        l.remove(0);
        root.left = rdeseriable(l);
        root.right = rdeseriable(l);
        
        return root;
    }
    
    public TreeNode deserialize(String data){
        String[] data_array = data.split(",");
        List<String> data_list = new LinkedList<>();
        return rdeserialize(data_list);
    }
}
```

### 二叉搜索树的最近公共祖先

给定一个二叉搜索树, 找到该树中两个指定节点的最近公共祖先。

百度百科中最近公共祖先的定义为：“对于有根树 T 的两个结点 p、q，最近公共祖先表示为一个结点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。”

**难点**：不知道如何入手，之前写过，但只有零碎的记忆

**祖先的定义**：若节点*p*在节点*root*的左（右）子树中，或者*p=root*，则称*root*是*p*的祖先。

**最近公共祖先的定义**：设节点*root*是*p*和*q*节点的某公共祖先，而且*root.left*和*root.right*不是*p*，*q*的公共祖先，此时*root*就是两个节点的公共祖先。

```java
class Solution{
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q){
        while(root != null){
            if((root.val < p.val)&&(root.val < q.val)){
                root = root.right;
            }else if((root.val>p.val) && (root.val < p.val)){
                root = root.left;
            }
            else break;            
        }
        return root;
    }
}
```

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution{
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p,TreeNode q){
        if(root == null) return root;
        if(p.val < root.val && q.val < root.val){
            return lowestCommonAncestor(root.left,p,q);
        }else if(p.val > root.val && q.val >root.val){
            return lowestCommonAncestor(root.right,p,q);
        }
        return root;
    }
}
```

### 不同的二叉搜索树||

给定一个整数n,生成所有由1...n为节点所组成的**二叉搜索树**

**示列**：

```java
输入：3
输出：
[
  [1,null,3,2],
  [3,2,null,1],
  [3,1,null,null,2],
  [2,1,3],
  [1,null,2,null,3]
]
解释：
以上的输出对应以下 5 种不同结构的二叉搜索树：

   1         3     3      2      1
    \       /     /      / \      \
     3     2     1      1   3      2
    /     /       \                 \
   2     1         2                 3
```

```java
class Solution{
    public LinkedList<TreeNode> generate_trees(int start, int end){
        LinkedList<TreeNode> all_trees = new LinkedList<TreeNode>();
        if(start > end){
            all_trees.add(null);
            return all_trees;
        }
        for(int i=start;i<=end;i++){
            LinkedList<TreeNode> left_trees = generate_trees(start,i-1);
            
            LinkedList<TreeNode> right_trees = generate_trees(i+1,end);
            
            for(TreeNode l : left_trees){
                for(TreeNode r : right_tress){
                    TreeNode current_tree = new TreeNode(i);
                    current_tree.left = l;
                    current_tree.right = r;
                    all_trees.add(current_tree);
                }
            }
        }
        return all_trees;
    }
    
    public List<TreeNode> generateTrees(int n){
        if(n == 0) {
            return new LinkedList<TreeNode>();
        }
        return generate_trees(1,n);
    }
}
```

### 填充每个节点的下一个右侧节点指针||

**层次遍历和辅助指针**

```java
/*
// Definition for a Node.
class Node {
    public int val;
    public Node left;
    public Node right;
    public Node next;

    public Node() {}
    
    public Node(int _val) {
        val = _val;
    }

    public Node(int _val, Node _left, Node _right, Node _next) {
        val = _val;
        left = _left;
        right = _right;
        next = _next;
    }
};
*/
class Solution{
    public Node connect(Node root){
        if(root == null){
            reutrn null;
        }
        Queue<Node> queue = new LinkedList<Node>();
        queue.add(root);
        while(!queue.isEmpty()){
            Node pre = null;
            int size = queue.size();
            for(int i=0;i<size;i++){
                Node cur = queue.poll();
                if(i>0){
                    pre.next = cur;
                }
                pre = cur;
                if(cur.left != null){
                    queue.offer(cur.left);
                }
                if(cur.right != null){
                    queue.offer(cur.right);
                }
            }
        }
        return root;
    }
}
```

利用N层为N+1层建立next指针。

```java
class Solution{
    Node prev, leftmost;
    public void processChild(Node childNode){
        if(childNode != null){
            if(this.prev != null){
                this.prev.next = childNode;
            }else{
                this.leftmost = childNode;
            }
        }
        this.prev = childNode;
    }
}

public Node connect(Node root){
    if(root == null){
        return root;
    }
    this.leftmost = root;
    
    Node curr = leftmost;
    
    while(this.leftmost != null){
        this.prev = null;
        curr = this.leftmost;
    }
}
```

### 剑指OFFER，树的子结构

输入两棵二叉树`A`和`B`，判断`B`是否是`A`的子结构。(约定空树不是任意一个树的子结构)

`B`是`A`的子结构， 即` A`中有出现和`B`相同的结构和节点值。

**思路**：先判断B树的根结点是否和A树的任意结点相等，再判断两者的左右结点是否相等。**但是怎样入手了**

首先先序遍历A树中的每个结点*n*<sub>*A*</sub>，再判断树A中以*n*<sub>*A*</sub>为根结点的子树是否包含树**B**。

**终止条件：**
当节点 B 为空：说明树 B已匹配完成（越过叶子节点），因此返回 true ；
当节点 A为空：说明已经越过树 AA 叶子节点，即匹配失败，返回 false ；
当节点 A 和 B 的值不同：说明匹配失败，返回 false ；
**返回值：**
判断 A 和 B 的左子节点是否相等，即 recur(A.left, B.left) ；
判断 A 和 B 的右子节点是否相等，即 recur(A.right, B.right) ；

```java
class Solution{
    public boolean isSubStructure(TreeNode A,TreeNode B){
        return (A != null && B != null) && (recur(A,B)||isSubStructure(A.left,B)||isSubStructure(A.right,B));
    }
    boolean recur(TreeNode A,TreeNode B){
        if(B == null) return true;
        if(A == null || A.val != B.val) return false;
        return recur(A.left,B.left)&&recur(A.right,B.right);
    }
}
```

- **时间复杂度**：`O(MN)`

- **空间复杂度**：O(M)

### 验证二叉搜索树

**思路**：中序遍历，判断当前结点是否大于中序遍历的前一个结点，如果大于，说明满足`BST`，继续遍历；否则直接返回`false`。

```java
class Solution{
    long pre = Long.MIN_VALUE;
    public boolean isValidBST(TreeNode root){
        if(root == null){
            return true;
        }
        if(!isValidBST(root.left)){
            return false;
        }
        if(root.val <= pre){
            return false;
        }
        pre = root.val;
        
        return isValidBST(root.right);
    }
}
```

### 二叉树的最小深度

给定一个二叉树，找出其最小深度。

最小深度是从根结点到最近叶子结点的最短路径上的结点数量。

**示例:**

给定二叉树 `[3,9,20,null,null,15,7]`,

```java
    3
   / \
  9  20
    /  \
   15   7
```

```java
class Solution{
    public int minDepth(TreeNode root){
        if(root == null) return 0;
        if((root.left == null) && (root.right == null)){
            return 1;
        }
        int min_depth = Integer.MAX_VALUE;
        if(root.left != null){
            min_depth = Math.min(minDepth(root.left),min_depth);
        }
        if(root.right != null){
            min_depth = Math.min(minDepth(root.right),min_depth);
        }
        return min_depth + 1;
    }
}
```

- 时间复杂度：我们访问每个节点一次，时间复杂度为 `O(N) `，其中 NN 是节点个数。
- 空间复杂度：最坏情况下，整棵树是非平衡的，例如每个节点都只有一个孩子，递归会调用 NN （树的高度）次，因此栈的空间开销是 `O(N)` 。但在最好情况下，树是完全平衡的，高度只有` log(N)`，因此在这种情况下空间复杂度只有 `O(log(N))`

### 二叉树的所有路径

**思路**：递归方式进行遍历，当到达叶子结点时，把结果添加到结果队列中，没到达时遍历左右左右子树。

```java
class Solution{
    public void construct_paths(TreeNode root ,String path, LinkedList<String> paths){
        if(root != null){
            path += Integer.toString(root.val);
            if((root.left == null) && (root.right == null)){
                paths.add(path);
            }else{
                path += "->";
                construct_paths(root.left,path,paths);
                construct_paths(root.right,path,paths);
            }
        }
    }
    
    public List<String> binaryTreePaths(TreeNode root){
        LinkedList<String> paths = new LinkedList<>();
        construct_pahts(root,"",paths);
        return paths;
    }
}
```

### 腐烂的橘子

在给定的网格中，每个单元格可以有以下三个值之一：

- 值 0 代表空单元格；
- 值 1 代表新鲜橘子；
- 值 2 代表腐烂的橘子。
- 每分钟，任何与腐烂的橘子（在 4 个正方向上）相邻的新鲜橘子都会腐烂。

返回直到单元格中没有新鲜橘子为止所必须经过的最小分钟数。如果不可能，返回 -1。

**思路**：此题可以被看做一个最短路径问题，求腐烂橘子到所有新鲜橘子的最短路径。利用一个队列存储腐烂橘子的坐标，每个坐标是用一个数组来存储的。

**难点**：一个变量在递归中怎样处理的。**定义成全局变量**。

```java
public int orangesRotting(int[][] grid){
    int M = grid.length;
    int N = grid[0].length;
    Queue<int[]> queue = new LinkedList<>();
    int count = 0;
    
    for(int i=0;i<M;i++){
        for(int j=0;j<N;j++){
            if(grid[i][j] == 1){
                count++;
            }
            if(grid[i][j] == 2){
                queue.add(new int[]{i,j});
            }
        }
    }
    
    int round = 0;
    while(count > 0 && !queue.isEmpty()){
        round++;
        int size = queue.size();
        for(int i=0;i<size;i++){
            int[] orange = queue.poll();
            int r = orange[0];
            int c = orange[1];
            
            if(r-1>=0 && grid[r-1][c] == 1){
                grid[r-1][c] = 2;
                count--;
                queue.add(new int[]{r-1,c});
            }
            
            if(r+1<M&&grid[r+1][c] == 1){
                grid[r+1][c] = 2;
                count--;
                queue.add(new int[]{r+1,c});
            }
            
            if(c-1>=0 && grid[r][c-1]==1){
                grid[r][c-1] = 2;
                count--;
                queue.add(new int[]{r,c-1});
            }
            
            if(c + 1 < N&&grid[r][c+1] == 1){
                grid[r][c+1] = 2;
                count--;
                queue.add(new int[]{r,c+1});
            }
        }
    }
    if(count > 0){
        return -1;
    }
    return round;
}
```

先求出有多少个好的橘子，保持初始时腐烂橘子的下标，并存入到队列中。

### 二叉树展开为链表

给定一个二叉树，原地将它展开成一个单链表。

例如，给定一个二叉树

```java
    1
   / \
  2   5
 / \   \
3   4   6
```

将其展开为

```java
1
 \
  2
   \
    3
     \
      4
       \
        5
         \
          6
```

**难点**：注意把二叉树的所有结点都添加到了右边，将**右子树添加到左子树最右节点处**。

```java
public void flatten(TreeNode root){
    while(root!=null){
        if(root.left == null){
            root = root.right;
        }else{
            TreeNode pre = root.left;
            while(pre.right != null){
                pre = pre.right;
            }
            pre.right = root.right;
            root.right = root.left;
            root.left = null;
            root = root.right;
        }
    }
}
```

### 判断二叉树是否平衡

**知识点**：树的深度等于左子树的深度与右子树的深度中的**最大值**+1

**思路**：后序遍历+剪枝。其实不太明白**剪枝是啥意思**。若判定某子树不是平衡树则 “剪枝” ，直接向上返回。

```java
class Solution{
    public boolean isBalanced(TreeNode root){
        return recur(root) != -1;
    }
    
    private int recur(TreeNode root){
        if(root == null) return 0;
        int left = recur(root.left);
        if(left == -1){
            return -1;
        }
        int right = recur(root.right);
        if(right == -1){
            return -1;
        }
        return Math.abs(left - right) < 2 ? Math.max(left,right) + 1:-1;
    }
}
```

### 不同的二叉搜索树

**思路：**动态规划

1. G(n)：长度n的序列的不同二叉搜索树的个数
2. F(i,n)：以i为根的不同二叉搜索树的个数(1 &lt;i&lt;n )

$$

$$

```java
public class Solution{
    public int numTrees(int n){
        int[] G = new int[n + 1];
        G[0] = 1;
        G[1] = 1;
        
        for(int i=2;i<=n;i++){
            for(int j=1;j<=i;j++){
                G[i] += G[j-1] * G[i-j];
            }
        }
        return G[n];
    }
}
```

### 分割链表

编写程序以x为基准分割链表，使得所有小于x的节点排在大于或等于x的节点之前。如果链表中包含了x，x只需出现在小于x的元素之后。分割元素x只需处于右半部分即可。其不需要被置于左右两部分之间。

**示例**：

```java
输入: head = 3->5->8->5->10->2->1, x = 5
输出: 3->1->2->10->5->5->8
```

**难点**：没看懂题目和给的示例。意思是比x小的的在比x大前面，不仅仅是在x的前面。

```java
public class ListNode{
    int val;
    ListNode next;
    ListNode(int x){
        val = x;
    }
}

class Solution{
    public ListNode partition(ListNode head,int x){
        if(head == null) return head;
        ListNode dummy = new ListNode(-1);
        dummy.next = head;
        ListNode pre = head;
        head = head.next;
        while(head != null){
            if(head.val < x){
                prev.next = head.next;
                head.next = dummy.next;
                dummy.next = head;
                head = prev.next;
            }else{
                prev=prev.next;
                head = head.next;
            }
        }
        return dummy.next;
    }
}
```

### 二叉树中的和为某一值的路径

**思路**：先序遍历 + 回溯

结果的判断条件：

```java
if(sum == 0 && root.left == null && root.right == null){
    paths.add(new LinkedList(path));
}
```

```java
class Solution{
    LinkedList<List<Integer>> res = new LinkedList<>();
    LinkedList<Integer> path = new LinkedList<>();
    
    public List<List<Integer>> pathSum(TreeNode root, int sum){
        recur(root,sum);
        return res;
    }
    
    private void recur(TreeNode root, int sum){
        if(root == null) return;
        res.add(root.val);
        sum -= root.val;
        if(sum == 0&&root.left == null&&root.right == null){
            res.add(new LinkedList(path));
        }
        recur(root.left,sum);
        recur(root.right,sum);
        // path.remove(path.size() - 1);
        path.removeLast();
    }
}
```

### 有序链表转换二叉搜索树

给定一个单链表，其中的元素按升序排序，将其转换为高度平衡的二叉搜索树。

本题中，一个高度平衡的二叉树是指一个二叉树的每个节点的左右子树的高度差的绝对差不超过1.

**思路**：给定列表中的中间元素将会作为二叉树的根，该点左侧的所有元素递归去构造左子树，同理右侧的元素构造右子树。

**难点**：怎样找到链表的中点，怎样结束递归

```java
class Solution{
    public TreeNode sortedListToBST(ListNode head){
        if(head == null) return head;
        ListNode mid = findMiddleElement(head);
        ListNode node = new ListNode(mid.val);
        
        if(head == mid){
            return node;
        }
        
        node.left = this.sortedListToBST(head);
        node.right = this.sortedListToBST(mid);
        return node;
    }
    
    private ListNode findMiddleElement(ListNode head){
        ListNode prevPtr = null;
        ListNode slowPtr = head;
        ListNode fastPtr = head;
        while(fastPtr != null && fastPtr.next != null){//attention
            prevPtr = slowPtr;
            slowPtr = slowPtr.next;
            fastPtr = fastPtr.next.next;
        }
        if(prevPtr != null){
            prevPtr.next = null;
        }
        return slowPtr;
    }
}
```

### 从链表中删除总和值为零的连续节点

给你一个链表的头节点 head，请你编写代码，反复删去链表中由 总和 值为 0 的连续节点组成的序列，直到不存在这样的序列为止。

删除完毕后，请你返回最终结果链表的头节点。

```java
class Solution{
    public ListNode removeZeroSumSublists(ListNode head){
		if(head.next == null){
            if(head.val == 0) return null;
            return head;
        }
        
        ListNode dummyHead = new ListNode(0);
        dummyHead.next = head;
        ListNode h = dummyHead;
        boolean flag = true;
        
        while(h.next != null){
            flag = true;
            ListNode p = h.next;
            if(p.val == 0){
                h.next = p.next;
                continue;
            }
            ListNode q = p.next;
            int sum = p.val;
            while(q != null){
                sum += q.val;
                if(sum == 0){
                    flag = false;
                    h.next = q.next;
                    break;
                }else{
                   	q = q.next; 
                }
            }
            if(flag) h = h.next;
        }
        return dummyHead.next;
    } 
}
```

**减少一个指针**

```java
class Solution{
    public ListNode removeZeroSumSublists(TreeNode head){
        ListNode dummy = new ListNode(0);
        ListNode h = dummy;
        h.next = head;
        ListNode q = h.next;
        while(h.next != null){
            int sum = 0;
            while(q!=null){
                sum += q.val;
                if(sum == 0){
                    break;
                }else{
                    q = q.next;
                }
            }
            if(sum == 0){
                q = q.next;
                h.next = q;
            }else{
                q = q.next;
                h = h.next;
            }
        }
        return dummy.next;
    }
}
```

### 复杂链表的深度拷贝

**思路**：HashMap实现，先对链表的值进行拷贝，之后对结点之间的指向关系进行拷贝。

```java
class Solution{
    public Node copyRandomList(Node head){
        HashMap<Node,Node> map = new HashMap<>();
        Node cur = head;
        
        while(cur != null){
            map.put(cur,new Node(cur.val));
            cur = cur.next;
        }
        
        cur = head;
        while(cur != null){
            map.get(cur).next = map.get(cur.next);
            map.get(cur).random = map.get(cur.random);
            cur = cur.next;
        }
        return map.get(head);
    }
}
```

### 寻找重复数组

1.数组中有一个重复的整数<==>链表中存在环

2.找到数组中的重复整数<==>找到链表的环入口

快慢指针如何走？

```xml
slow = slow.next ==> slow = nums[slow]
fast = fast.next.next ==> fast = nums[nums[fast]]
```

```java
class Solution{
    public int findDuplicate(int[] nums){
        int slow = 0;
        int fast = 0;
        slow = nums[slow];
        fast = nums[nums[fast]];
        while(slow != fast){
            slow = nums[slow];
            fast = nums[nums[fast]];
        }
        int pre1 = 0;
        int pre2 = slow;
        while(pre1!= pre2){
            pre1 = nums[pre1];
            pre2 = nums[pre2];
        }
        return pre1;
    }
}
```

### 删除排序链表中的重复元素||

双指针解决

**难点**：之前总是在想怎样把链表连起来，只需要改变两个节点的指向就可以了

```java
class Solution{
    public ListNode deleteDuplicates(ListNode head){
        if(head == null) return null;
        if(head.next == null) reutrn head;
        ListNode dummy = new ListNode(0);
        ListNode slow = dummy;
        slow.next = head;
        ListNode fast = head;
        while(fast != null){
            if((fast.next != null && fast.val != fast.next.val)||fast.next == null){
                if(slow.next == fast){
                    slow = fast;
                }else{
                    slow.next = fast.next;
                }
            }
            fast = fast.next;
        }
        return dummy.next;
    }
}
```

### 环形链表

双指针方法：

**思路**：设置快指针和慢指针，当第一次相遇的时候，快指针走过的路径f是慢指针走过的路径s的两倍f=2s，此时快指针比慢指针走多走过nb结点，其中n是环的个数，b是环有多少个结点，f=s+nb两个关系相减f=2nb,s=nb。前提知道的是a+nb一定是环入口。所以只要让s再走a个就行，此时再用一个指针从头开始，相遇的地方就是环的入口。

```java
public class Solution{
    public ListNode detectCycle(ListNode head){
        ListNode fast = head, slow = head;
        while(true){
            if(fast == null || fast.next == null) return null;
            fast = fast.next.next;
            slow = slow.next;
            if(fast == slow) break;
        }
        fast = head;
        while(fast != slow){
            fast = fast.next;
            slow = slow.next;
        }
        return slow;
    }
}
```

### LRU缓存机制

运用你所掌握的数据结构，设计和实现一个LRU（最近最少使用）缓存机制。应该支持一下参数：获取数据`get`和写入数据`put`。

获取数据 get(key) - 如果关键字 (key) 存在于缓存中，则获取关键字的值（总是正数），否则返回 -1。
写入数据 put(key, value) - 如果关键字已经存在，则变更其数据值；如果关键字不存在，则插入该组「关键字/值」。当缓存容量达到上限时，它应该在写入新数据之前删除最久未使用的数据值，从而为新的数据值留出空间。



```java
class LRUCache {

    public LRUCache(int capacity) {

    }
    
    public int get(int key) {

    }
    
    public void put(int key, int value) {

    }
}
```

哈希表查找快，但是数据无固定顺序；链表有顺序之分，插入删除快，但是查找慢。哈希链表实现LRU。

LRU缓存算法的核心数据结构就是哈希链表，双向链表和哈希表的结合体。

![HashLinkedList](https://pic.leetcode-cn.com/9201fabe4dfdb5a874b43c325d39857182c8ec267f830649a52dda90a63d6671-file_1562356927818)



先实现双向链表

```java
//结点
public class Node{
    public int key,val;
    public Node next,prev;
    public Node(int k,int v){
        this.key = k;
        this.val = v;
    }
}

//建立双向链表
class DoubleList{
    private Node head,tail;//头尾虚节点
    private int size;
    
    public DoubleList(){
        head = new Node(0,0);
        tail = new Node(0,0);
        size = 0;
    }
    
    public void addFirst(Node x){
        x.next = head.next;
        x.prev = head;
        head.next.prev = x;
        tail.prev.next = x;
        size++;
    }
    
    public void remove(Node x){
        head.next = x.next;
        tail.prev = x.prev;
        size--;
    }
    
    //删除链表的最后一个结点
    public Node removeLast(){
        if(tail.prev == head){
            return null;
        }else{
            Node last = tail.prev;
            remove(last);
            return last;
        }
    }
    
    public int size(){
        return size;
    }
}

// LRU实现
class LRUCache{
    //key -> Node(key,val)
    private HashMap<Integer,Node> map;
    //Node(k1,v1)<->Node(k2,v2)
    private DoubleList cache;
    //最大容量
    private int cap;
    
    public LRUCache(int capacity){
        this.cap = capacity;
        map = new HashMap<>();
        cache = new DoubleList();
    }
    
    public int get(int key){
        if(!map.containsKey(key))
            return -1;
        int val = map.get(key).val;
        //利用put方法把该数据提前
        put(key,val);
        return val;
    }
    
    public void put(int key,int val){
        Node x = new Node(key,val);
        
        if(map.containsKey(key)){
            cache.remove(map.get(key));
            cache.addFirst(x);
            //更新map中对应的数据
            map.put(key,x);
        }else{
            if(cap == cache.size()){
                Node last = cache.removeLast();
                map.remove(last.key);
            }
            //直接添加到头部
            cache.addFirst(x);
            map.put(Key,x);
        }
    }
}
```

### 字符串相加

给定两个字符串形式的非负整数 `num1` 和`num2` ，计算它们的和。

**注意：**

`num1` 和`num2` 的长度都小于 `5100`.
`num1` 和`num2 `都只包含数字 `0-9`.
`num1` 和`num2` 都不包含任何前导零。
你不能使用任何內建 `BigInteger` 库， 也不能直接将输入的字符串转换为整数形式。

```java
class Solution{
    public String addString(String num1,String num2){
		StringBuilder res = new StringBuilder();
        int i = nums1.length() - 1;
        int j = nums2.length() - 1, carry = 0;
		whille(i >= 0 ||j >= 0){
            int n1 = i >= 0?num1.charAt(i) - '0':0;
            int n2 = j >= 0?num2.charAt(j) - '0':0;
            
            int temp = n1 + n2 + carry;
            carry = temp / 10;
            res.append(temp % 10);
            i--,j--;
        }
        if(carry == 1){
            res.append(1);
        }
        return res.reverse().toString();
    }
}
```

### 二叉树的右视图

按照根结点->右子树->左子树的顺序访问。

```java
class Solution{
    List<Integer> res = new ArrayList<>();
    
    public List<Integer> rightSideView(TreeNode root){
        dfs(root,0);
        return res;
    }
    
    private void dfs(TreeNode root,int depth){
        if(root == null) return;
        if(depth == res.size()){
            res.add(root.val);
        }
        depth++;
        dfs(root.right,depth);
        dfs(root.left,depth);
    }
}
```

**时间复杂度**：O(N)，每个结点都访问了一次

**空间复杂度**：O(N)，深度是logN，但是可能会退化成一条链表，深度就是N

### 路径总和||

给定一个二叉树和一个目标和，找到所有从根节点到叶子节点路径总和等于给定目标和的路径。

**说明:** 叶子节点是指没有子节点的节点。

```java
class Solution{
    public List<List<Integer>> pathSum(TreeNode root,int sum){
        List<List<Integer>> paths = new ArrayList<>();
        List<Integer> path = new ArrayList<>();
        
        go(root,0,sum,path,paths);
        return paths;
    }
    
    private void go(TreeNode root,int sum, int target,ArrayList<Integer> path, List<List<Integer>> paths){
        if(root == null) return;
        sum += root.val;
        path.add(node.val);
        if(sum == target && node.left == null && node.right == null){
            paths.add(new ArrayList<Integer>(path));
        }else{
            go(root.left,sum,target,path,paths);
            go(root.right,sum,target,path,paths);
        }
        path.remove(path.size() - 1);
    }
}
```

### 无重复的最长子字符串

**思路**：滑动窗口，set

```java
class Solution{
    public int lengthOfLengestSubstring(String s){
        HashSet<Character> set = new HashSet<>();
        int n = s.length();
        int rk = 0, ans = 0;
        for(int i=0;i<n;i++){
            if(i!=0){
                set.remove(s.charAt(i-1));
            }
            while(rk<n&&!set.contains(s.charAt(rk))){
                set.add(s.charAt(rk));
                rk++;
            }
            ans = Math.max(ans,rk-i);
        }
        return ans;
    }
}
```

**时间复杂度**：$O（N）$

**空间复杂度**：$O(|E|)$，这里的$E$表示字符集，字符串可以出现的字符。

### 长度最小的子数组-滑动窗口的变种

给定一个含有 n 个正整数的数组和一个正整数 s ，找出该数组中满足其和 ≥ s 的长度最小的 连续 子数组，并返回其长度。如果不存在符合条件的子数组，返回 0。

**方法使用一个叫双指针**：

```java
class Solution{
    public int minSubArrayLen(int s,int[] nums){
        int n = nums.length;
        if(n == 0) return 0;
        int ans = Integer.MAX_VALUE;
        int start = 0, end = 0;
        int sum = 0;
        while(end < n){
            sum += nums[end];
            while(sum >= s){
                ans = Math.min(ans,end - start+1);
                sum -= nums[start];
                start++;
            }
            end++;
        }
        return ans = Integer.MAX_VALUE?0:ans;
    }
}
```

### 字符串的排列

给定两个字符串 **s1** 和 **s2**，写一个函数来判断 **s2** 是否包含 **s1** 的排列。

换句话说，第一个字符串的排列之一是第二个字符串的子串。

```java
class Solution{
    public boolean checkInclusion(String s1,String s2){
        int len1 = s1.length(), len2 = s2.length();
        if(len1 > len2) return false;
        int[] ch_count1 = new int[26],ch_count2 = new int[26];
        for(int i=0;i<len1;i++){
            ch_count1[s1.charAt(i) - 'a']++;
            ch_count2[s2.charAt(i) - 'a']++;
        }
        for(int i=len1;i<len2;i++){
            if(isEqual(ch_count1,ch_count2)) return true;
            ch_count2[s2.charAt(i-len1) - 'a']--;
            ch_count2[s2.charAt(i) - 'a']++;
        }
        return isEqual(ch_count1,ch_count2);
    }
    private boolean isEqual(int[] ch_count1,int[] ch_count2){
        for(int i=0;i<26;i++){
            if(ch_count1[i] != ch_count2[i]){
                return false;
            }
        }
        return true;
    }
}
```



### 两数相加

```java
class Solution{
    public LisNode addTwoNumbers(ListNode l1, ListNode l2){
        ListNode dummy = new ListNode(0);
        ListNode p = l1, q = l2;
        ListNode curr = dummy;
        int carry = 0;
        while(p != null || q != null){
            int n1 = p != null ? p.val:0;
            int n2 = q != null ? q.val:0;
            int sum = n1 + n2 + carry;
            carry = sum / 10;
            ListNode temp_Node = new ListNode(sum % 10);
            curr.next = temp_Node;
            curr = curr.next;
            
            if(p != null) p = p.next;
            if(q != null) q = q.next;
        }
        if(carry >0){
            curr.next = new ListNode(carry);
        }
        return dummy.next;
    }
}
```

### 二叉树的完全性检验

**思路**：广度优先搜索，**表示深度和位置的方法**，用1表示根结点，对于任意一个结点V，他的左孩子为2*v，右孩子为2*v+1.

```java
class Solution{
    public boolean isCompleteTree(TreeNode root){
        List<ANode> nodes = new ArrayList<>();
        nodes.add(new ANode(root,1));
        int i=0;
        while(i < nodes.size()){
            ANode anode = nodes.get(i++);
            if(anode.node != null){
                nodes.add(new ANode(anode.node.left,anode.code * 2));
                nodes.add(new ANode(anode.node.right,anode.code * 2 + 1));
            }
        }
        return nodes.get(i-1).code == nodes.size();
    }
}

class ANode{
    TreeNode node;
    int code;
    ANode(TreeNode node, int code){
        this.node = node;
        this.code = code;
    }
}
```

### 剑指Offer. 连续子数组的最大和

输入一个整型数组，数组里有正数也有负数。数组中的一个或连续多个整数组成一个子数组。求所有子数组的和的最大值。

要求时间复杂度为O(n)。

**思路**：

- 当 $dp[i - 1] > 0$时：执行 $dp[i] = dp[i-1] + nums[i]$；
- 当 $dp[i−1]≤0$ 时：执行 $dp[i] = nums[i]$ ；

```java
class Solution{
    public int maxSubArray(int[] nums){
        int res = nums[0];
        for(int i=1;i<nums.length;i++){
            nums[i] += Math.max(nums[i-1],0);
            res = Math.max(res,nums[i]);
        }
        return res;
    }
}
```

```java
class Solution{
    public int maxSubArray(int[] nums){
        int max = nums[0];
        int former = 0;
        int cur = nums[0];
        for(int num :nums){
            cur = num;
            if(former>0) cur+=former;
            if(cur > max) max = cur;
            former = cur;
        }
        return max;
    }
}
```

### 前K个高频元素

最小堆，可以利用堆这种数据结构，对k频率之后的元素不再进行处理，进一步优化时间复杂度。

- 利用HashMap来建立数字和其出现次数的映射，通过遍历一遍数组统计元素的频率
- 维护一个元素数目为k的最小堆
- 把每次新的元素和堆顶元素进行比较，如果新元素的频率大于堆顶元素时，则弹出堆顶元素，添加新元素进堆。
- 最后就只剩下只有k个元素了

```java
class Solution{
    public List<Integer> topKFrequent(int[] nums,int k){
        HashMap<Integer> map = new HashMap<>();
        for(int num : nums){
            if(map.containsKey(num)){
                map.put(num,map.get(num) + 1);
            }else{
                map.put(num,1);
            }
        }
        
        //利用优先队列实现最小堆
        PriorityQueue<Integer> pq = new PriorityQueue<>(new Comparator<Integer>(){
            @Override
            public int compare(Integer a, Integer b){
                return map.get(a) - map.get(b);
            }
        });
        for(Integer key:map.keySet()){
            if(pq.size() < k){
                pq.add(key);
            }else if(map.get(key) > map.get(pq.peek())){
                pq.remove();
                pq.add(key);
            }
        }
        List<Integer> res = new ArrayList<>();
        while(!pq.isEmpty()){
            res.add(pq.remove());
        }
        return res;
    }
}
```

### 对称二叉树

递归遍历

```java
class Solution{
    public boolean isSymmetric(TreeNode root){
        if(root == null) return true;
        return helper(root,root);
    }
    private boolean helper(TreeNode node1, TreeNode node2){
        if(node1== null && node2 == null) return true;
        if(node1 == null || node2 == null) return false;
        return node1.val == node2.val && (helper(node1.left,node2.right)) && (helper(node1.right, node2.left));
    }
}
```

### 段氏回文

### 有效的括号

map中为什么要多放一组'?'。

因为此时在stack为空且c为右括号时，可以正常提前返回false

```java
class Solution{
    private static final Map<Character, Character> map = new HashMap<>(){
        {
            put('{','}');put('[',']');put('(',')');
        }
    };
    
    public boolean isValid(String s){
        if(s.length()>0&&(!map.containsKey(s.charAt(0)))) return false;
        Stack<Character> stack = new Stack<>();
        for(Character c : s.toCharArray()){
            if(map.containsKey(c)) stack.push(c);
            else if(stack.isEmpty() || map.get(stack.pop()) != c){
                return false;
            }
        }
        return stack.isEmpty();
    }
}
```

### 移动零

给定一个数组 `nums`，编写一个函数将所有 `0` 移动到数组的末尾，同时保持非零元素的相对顺序。

```java
class Solution{
    public void moveZeros(int[] nums){
        if(nums == null) return;
        int j=0;
        for(int i=0;i<nums.length;i++){
            if(nums[i]!=0){
                nums[j++]=nums[i];
            }
        }
        for(int i=j;i<nums.length;i++){
            nums[i] = 0;
        }
    }
}
```

### 链表的中间节点

快慢指针

```java
class Solution {
    public ListNode middleNode(ListNode head) {
        ListNode fast = head;
        ListNode slow = head;
        while(fast != null && fast.next != null){
            fast = fast.next.next;
            slow = slow.next;
        }
        return slow;
    }
}
```

### 反转链表

**思路**：递归，调整指向

```java
class Solution{
    public ListNode reverseList(ListNode head){
        if(head == null || head.next == null) return head;
        
        ListNode newList = reverseList(head.next);
        
		head.next.next = head;
        head.next = null;
        return newList;
    }
}
```

### 相同的树

解法和对称树一样

```java
class Solution {
    public boolean isSameTree(TreeNode p, TreeNode q) {
        if(p == null && q == null) return true;
        return isSame(p,q);
    }
    private boolean isSame(TreeNode node1, TreeNode node2){
        if(node1 == null && node2 == null) return true;
        if(node1 == null || node2 == null) return false;
        return node1.val == node2.val && (isSame(node1.left, node2.left)) && (isSame(node1.right, node2.right));
    }
}
```

### X的平方根

**思路**：二分法，每次取半，取右中位数

**方法1**

```java
public class Solution{
    public int mySqrt(int x){
        if(x == 0) return 0;
        
        long left = 1;
        long right = x / 2;
        while(left < right){
            long mid = (left + right + 1) >>> 1;
            long square = mid * mid;
            if(square > x){
                right = mid - 1;
            }else{
                left = mid;
            }
        }
        return (int)left;
    }
}
```

**方法二**

```java
public class Solution{
    public int mySqrt(int a){
        long x = a;
        while(x * x > a){
            x = (x + a / x) / 2;
        }
        return (int)x;
    }
}
```

### 二叉树的最大路径和

**思路**：这种题看到答案很好懂，自己想不出来，利用了一个全局变量来维护最大路径和；节点的最大路径和取决于该节点的值与该节点的左右子节点的最大贡献值。

```java
class Solution{
    int max_sum = Integer.MIN_VALUE;
    
    public int maxPathSum(TreeNode root){
        maxGain(root);
        return maxSum;
    }
    
    private int maxGain(TreeNode node){
        if(node == null) return 0;
        
        int leftGain = Math.max(maxGain(node.left),0);
        int rightGain = Math.max(maxGain(node.right),0);
        
        int priceNewpath = node.val + leftGain +rightGain;
        maxSum = Math.max(max_sum, priceNewpath);
        
        return node.val + Math.max(leftGain,rightGain);
    }
}
```

### 旋转数组的最小数字

**思路**：二分法

```java
class Solution{
    public int minArray(int[] nums){
        int i=0, j=nums.length-1;
        while(i<j){
            int mid = (i + j) / 2;
            if(nums[mid] > nums[j]){
                i = mid + 1;
            }else if(nums[mid] < nums[j]){
                j = mid;
            }else{
                j--;
            }
        }
        return nums[i];
    }
}
```

### 相交链表

```java
public ListNode getIntersectionNode(ListNode head1,ListNode head2){
    if(head1 == null || head2 == null){
        return null;
    }
    ListNode p1 = head1, p2 = head2;
    while(p1 != p2){
        p1 = p1 == null ? head2:p1.next;
        p2 = p2 == null ? head1:p2.next;
    }
    return p1;
}
```

### 三数之和

```java
class Solution{
    public static List<List<Integer>> threeSum(int[] nums){
        List<List<Integer>> ans = new ArrayList<>();
        if(nums == null || nums.length < 3) return ans;
        Array.sort(nums);
        int len = nums.length;
        for(int i=0;i<len;i++){
            if(nums[i] > 0) break;
            if(i > 0 && nums[i] == nums[i-1]) continue;
            int L = i+1;
            int R = len - 1;
            while(L < R){
                int sum = nums[i] + nums[L] + nums[R];
                if(sum == 0) {
                    ans.add(Arrays.asList(nums[i],nums[L],nums[R]));
                    while(L < R && nums[L] == nums[L+1]) L++;
                    while(L < R && nums[R] == nums[R-1]) R--;
                    L++;
                    R--;
                }else if(sum < 0) L++;
                else if(sum > 0) R--;
            }
        }
        return ans;
    }
}
```

### 重排链表

**思路：**利用列表存储，采用双指针

```java
public void reorderList(ListNode head){
    if(head == null) return;
    
    List<ListNode> list = new ArrayList<>();
    while(head!=null){
        list.add(head);
        head = head.next;
    }
    int i=0, j=list.size()-1;
    while(i<j){
        list.get(i).next = list.get(j);
        i++;
        if(i == j) break;
        list.get(j).next = list.get(i);
        j--;
        
    }
    list.get(i).next = null;
}
```

**解法2**：递归，结束条件分别对应奇数和偶数，即一个结点和2个结点

```java
public void reorderList(ListNode head){
    if(head == null) return;
    
    int len = 0;
    ListNode h = head;
    while(h!=null){
        len++;
        h=h.next;
    }
    helper(head,len);
}
private ListNode helper(ListNode head, int len){
    if(len == 1){
        ListNode outTail = head.next;
        head.next = null;
        return outTail;
    }
    if(len == 2){
        ListNode outTail = head.next.next;
        head.next.next = null;
        return outTail;
    }
    
    //得到对应的尾结点，交换指针
    ListNode tail = helper(head.next,len-2);
    ListNode subHead = head.next;
    head.next = tail;
    ListNode outTail = tail.next;
    tail.next = subHead;
    return outTail;
}
```

### 字符串转换整数

```java
class Solution{
    public int myAtoi(String str){
		if(str == null || str.length() == 0) return 0;
        int i=0;
        int ans = 0;
        boolean if_Fu = false;
        while(i<str.length()&&str.charAt(i)==' ') i++;
        if(i<str.length() && (str.charAt(i) == '-' || str.charAt(i) == '+')){
            if(str.charAt(i) == '-') if_Fu = true;
            i++;
        }
        for(;i<str.length()&&Character.isDigit(str.charAt(i));i++){
            int digit = str.charAt(i) -'0';
            
            if(ans > (Integer.MAX_VALUE - digit) / 10){
                return if_Fu == true?Integer.MIN_VALUE:Integer.MAX_VALUE;
            }
            ans = ans * 10 + digit;
        }
        return if_Fu?-ans:ans;
    }
}
```

### 将数组拆分成斐波那契序列

```java
class Solution{
    public List<Integer> splitIntoFibonacci(String s){
        int N = s.length();
        for(int i=0;i<Math.min(10,N);i++){
            if(s.charAt(0) == '0' && i > 0) break;
            long a = Long.valueOf(s.substring(0,i+1));
            if(a >= Integer.MAX_VALUE) break;
            
            search: for(int j=i+1;j<Math.min(i+10,N);++j){
                if(s.charAt(i+1)=='0'&& j > i+1) break;
                long b = Long.valueOf(s.substring(i+1,j+1));
                if(b >= Integer.MAX_VALUE) break;
                
                List<Integer> fib = new ArrayList<>();
                fib.add((int) a);
                fib.add((int) b);
                
                int k = j + 1;
                while(k < N){
                    long nxt = fib.get(fib.size() - 2) + fib.get(fib.size() - 1);
                    String nxts = String.valuOf(nxt);
                    
                    if(nxt <= Integer.MAX_VALUE && s.substring(k).startsWith(nxts)){
                        k += nxts.length();
                        fib.add((int)nxt);
                    }else{
                        continue search;
                    }
                }
                if(fib.size() >= 3) return fib;
            }
        }
        return new ArrayList<Integer>();
    }
}
```

### 缺失的第一个正数

给你一个为排序的整数数组，请找出其中没有出现的第一个正整数。

对数据进行遍历，对于遍历到的数$x$，如果他在$[1,N]$的范围内，那么就将数组中的第$x-1$个位置打上标记。在遍历结束之后，如果所以的位置都被打上了标记，那么答案是$N+1$，否则答案是最小的没有打上标签的位置加1。

```java
class Solution{
    public int firstMissingPositive(int[] nums){
        int n = nums.length;
        for(int i=0;i<n;++i){
            if(nums[i] <= 0) nums[i] = 1 + n;
        }
        for(int i=0;i<n;i++){
			int num = Math.abs(nums[i]);
            if(num <= n){
                nums[num - 1] = -Math.abs(nums[num - 1]);
            }
        }
        for(int i=0;i<n;i++){
            if(nums[i] > 0) return i+1;
        }
        return n+1;
    }
}
```

### 二分查找

```java
class Solution{
    public int search(int[] nums, int target){
        int i=0;
        int j=nums.length - 1;
        while(i<=j){
            int mid = (i+j) / 2;
            if(nums[mid] == target) return mid;
            if(nums[mid]>target){
                j = mid - 1;
            }else{
                i = mid + 1;
            }
        }
        return -1;
    }
}
```

### 最长的连续子序列（hard）

给定一个未排序的整数数组，找出最长连续序列的长度。时间复杂度为O(n)。

**难点**：怎样判断连续？？？用HashSet不断尝试和匹配$x+1,x+2$,...是否存在

```java
class Solution{
    public int longestConsecutive(int[] nums){
        Set<Integer> nums_set = new HashSet<>();
        for(int num : nums){
            nums_set.add(num);
        }
        int longestStreak = 0;
        
        for(int num:nums_set){
            if(!nums_set.contains(num - 1)){
                int currentNum = num;
                int currentStreak = 1;
                
                while(nums_set.contains(currentNum + 1)){
                    currentNum += 1;
                    currentStreak += 1;
                }
                longestStreak = Math.max(longestStreak, currentStreak);
            }
        }
        return longestStreak;
    }
}
```

### 上题的演变--最长上升子序列（已审核）

给定一个无序的整数数组，找到其中最长上升子序列的长度。时间复杂度$O(nlogn)$

```xml
输入: [10,9,2,5,3,7,101,18]
输出: 4 
解释: 最长的上升子序列是 [2,3,7,101]，它的长度是 4。
```

**思路**：动态规划

```java
public class Solution{
    public int lengthOfLIS(int[] nums){
        if(nums.length == 0) return 0;
        int[] dp = new int[nums.length];
        dp[0] = 1;
        int maxans = 1;
        for(int i=1;i<dp.length;i++){
            int maxval = 0;
            for(int j=0;j<i;j++){
                if(nums[i] > nums[j]){
                    maxval = Math.max(maxval, dp[j]);
                }
            }
            dp[i] = maxval + 1;
            maxans = Math.max(maxans,dp[i]);
        }
        return maxans;
    }
}
```

### 演变--最长连续递增序列

给定一个未经排序的整数数组，找到最长且**连续**的的递增序列，并返回该序列的长度。

```xml
输入: [1,3,5,4,7]
输出: 3
解释: 最长连续递增序列是 [1,3,5], 长度为3。
尽管 [1,3,5,7] 也是升序的子序列, 但它不是连续的，因为5和7在原数组里被4隔开。 
```

**思路**：每个（连续）增加的子序列是不相交的，并且每当 `nums[i-1]>=nums[i]` 时，每个此类子序列的边界都会出现。

```java
class Solution{
    public int findLengthOfLCIS(int[] nums){
		int ans = 0, anchor = 0;
        for(int i=0;i<nums.length;i++){
            if(i > 0 && nums[i-1] >= nums[i]) anchor = i;
            ans = Math.max(ans,i-anchor+1);
        }
    }
}
```

**动态规划**

```java
class Solution{
    public int findLengthOfLCIS(int[] nums){
        if(nums == null) return 0;
        int n = nums.length;
        int ans = 1;
        int[] dp = new int[n];
        dp[0] = 1;
        for(int i=1;i<n;i++){
            if(nums[i] > nums[i-1]) dp[i]=dp[i-1]+1;
            res = Math.max(ans,dp[i]);
        }
        return res;
    }
}
```

### 三线程按顺序交替打印ABC的四中方法

```java
public class ABC_Lock {
    private static Lock lock = new ReentrantLock();// 通过JDK5中的Lock锁来保证线程的访问的互斥
    private static int state = 0;//通过state的值来确定是否打印
    static class ThreadA extends Thread {
        @Override
        public void run() {
            for (int i = 0; i < 10;) {
                try {
                    lock.lock();
                    while (state % 3 == 0) {// 多线程并发，不能用if，必须用循环测试等待条件，避免虚假唤醒
                        System.out.print("A");
                        state++;
                        i++;
                    }
                } finally {
                    lock.unlock();// unlock()操作必须放在finally块中
                }
            }
        }
    }
    static class ThreadB extends Thread {
        @Override
        public void run() {
            for (int i = 0; i < 10;) {
                try {
                    lock.lock();
                    while (state % 3 == 1) {
                        System.out.print("B");
                        state++;
                        i++;
                    }
                } finally {
                    lock.unlock();// unlock()操作必须放在finally块中
                }
            }
        }
    }
    static class ThreadC extends Thread {
        @Override
        public void run() {
            for (int i = 0; i < 10;) {
                try {
                    lock.lock();
                    while (state % 3 == 2) {
                        System.out.print("C");
                        state++;
                        i++;
                    }
                } finally {
                    lock.unlock();// unlock()操作必须放在finally块中
                }
            }
        }
    }
    public static void main(String[] args) {
        new ThreadA().start();
        new ThreadB().start();
        new ThreadC().start();
    }
}
```

### 二叉树的直径



### 二叉树的最近公共祖先

**思路**：通过递归对二叉树进行后续遍历，当遇到$p$或$q$时返回。从底至顶回溯，当及节点$p$,$q$在节点root的异侧时，节点$root$即为最近公共祖先，则向上返回$root$。

```java
class Solution{
    public TreeNode lowestCommonAncestor(TreeNode root,TreeNode p,TreeNode q){
		if(root == null || p == root || q == root){
            return root;
        }
        TreeNode left = lowestCommonAncestor(root.left,p,q);
        TreeNode right = lowestCommonAncestor(root.right,p,q);
        
        if(left == null) return right;
        if(right == null) return left;
        return root;
    }
}
```

### 二叉搜索树的最近公共祖先

```java
class Solution{
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q){
        if(root == null) return null;
        if(p.val > root.val && q.val > root.val){
                return lowestCommonAncestor(root.right,p,q);
            }
            else if(p.val < root.val && q.val < root.val){
                 return lowestCommonAncestor(root.left,p,q);
            }else{
                return root;
            }
    }
}
```

**迭代**

```java
class Solution{
    public TreeNode lowestCommonAncestor(TreeNode root,TreeNode p,TreeNode q){
        while(root != null){
            if(root.val > p.val && root.val > q.val){
                root = root.left;
            }else if(p.val > root.val && q.val > root.val){
                root = root.right;
            }else{
                break;
            }
        }
        return root;
    }
}
```

### 求1+2+。。。+n

**思路**：利用与&&的短路效应

```java
class Solution{
    int res = 0;
    public int sumNums(int n){
        boolean x = n > 1 && sumNums(n-1)>0; //当x等于1时，不满足n > 1
        res += n;
        return res;   
    }
}
```

### 股票的最大利润

找到最小值，之后的每个值与之作减法，保存差的最大值。

```java

```

### 约瑟夫环问题

```java
class Solution{
    public int lastRemaining(int n,int m){
        int ans = 0;
        for(int i=2;i<=n;i++){
            ans = (ans + m) % i;
        }
        return ans;
    }
}
```

### 二叉平衡树

```java
class Solution{
    boolean flag = true;
    public boolean isBalanced(TreeNode root){
        if(root == null) return true;
        helper(root);
        return false;
    }
    
    private void helper(TreeNode root){
        if(root == null) return 0;
        
        int left = helper(root.left);
        int right = helper(root.right);
        
        if(Math.abs(left - right) > 1){
            flag = false;
        }
        return Math.max(left,right) + 1;
    }
}
```

### 二叉树的直径

**思路**：深度优先搜索，每个节点的深度，用一个局部变量来保存

```java
class Solution{
    int max = 0;
    public int diameterOfBinaryTree(TreeNode root){
        depth(root);
        return max;
    }
    
    public int depth(TreeNode root){
        if(root == null) return 0;
        
        int L = depth(root.left);
        int R = depth(root.right);
        
        max = Math.max(max, L + R);
        
        return Math.max(L,R) + 1;
    }
}
```

### 合并两个有序的链表

```java
class ListNode{
    int val;
    ListNode next;
    ListNode(int x){ val = x;}
}

public class Main {

    public ListNode mergeTwoLists(ListNode l1,ListNode l2){
        if(l1 == null && l2 == null) return null;
        ListNode dummy = new ListNode(0);
        ListNode cur = dummy;
        while(l1!=null&&l2 != null){
            if(l1.val < l2.val){
                cur.next = l1;
                l1 = l1.next;
            }else {
                cur.next = l2;
                l2 = l2.next;
            }
            cur = cur.next;
        }
        if(l1!=null) cur.next = l1;
        if (l2!=null) cur.next = l2;
        return dummy.next;
    }
}
```

### 翻转字符串里的单词



### 数字范围按位与

给定范围[m，n]，其中0 <= m <= n <= 2147483647，返回此范围内所有数字的按位与（包含 m, n 两端点）。

**示例 1:** 

```xml
输入: [5,7]
输出: 4
```

**思路**：向右移位，左右两数相等时，退出循环，并向左移位相同的次数。

```java
class Solution{
    public int rangeBitwiseAnd(int m, int n){
        int shift = 0;
        while(m < n){
            m >>= 1;
            n >>= 1;
            shift++;
        }
        return m<<shift;
    }
}
```

### k个一组翻转链表

**难点**：怎样确定到k个一组？每一组又是怎样翻转的？





**可重入锁**：是指可重复可递归调用的锁，在外层使用锁之后，在内层任然可以使用，并且不会发生死锁。

**Arrays.Sort()，传入一个Comparator实例**

```java
Arrays.sort(array,new Comparator<String>(){
    public int compare(String s1, String s2){
        return s1.compareTo(s2);
    }
});
```

**写一个单例模式**

```java
public class Singleton{
    private static final Singleton INSTANCE = new Singleton();
    
    private static Singleton getInstance(){
        return INSTANCE;
    }
    
    private Singleton(){
        
    }
}

//调用
Singleton instance = Singleton.getInstance();
```

**使用迭代器遍历集合**

```java
for(Iterator<String> it = list.iterator();it.hasNext();){
    it.next();
}
```

**优先队列--定义优先级**

```java
PriorityQueue<String> pq = new PriorityQueue<String>((s1,s2)->{
    return s2.compareTo(s1);
});

//Comparator
PriorityQueue<String> pq = new PriorityQueue<>(new Comparator<String>() {
	@Override
    public int compare(String o1, String o2) {
        return o2.compareTo(o1);
    }
});
```

### 最大子序和

给定一个整数数组 `nums` ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。

```java
class Solution{
    public int maxSubArray(int[] nums){
        int pre = 0, maxVal = nums[0];
        for(int num : nums){
            pre = Math.max(pre + num, num);
            maxVal = Math.max(pre, maxVal);
        }
        return maxVal;
    }
}
```

动态规划中包含3个重要的概念：

- 最优子结构
- 边界
- 状态转移公式

以跳台阶为例，最优子结构为f(10)=f(9) + f(8)，边界是f(1)=1, f(2)=2，状态转移公式f(n)=f(n-1) + f(n-2)

### 数组中的第K个最大元素

```java
public class Solution{
    public int findKthLargest(int[] nums, int k){
        int len = nums.length;
        int left = 0;
        int right = len - 1;
        
        int target = len - k;
        
        while(true){
            int index = partition(nums,left, right);
            if(index == target){
                return nums[index];
            }else if(index < target){
                left = index + 1;
            }else{
                right = index - 1;
            }
        }
    }
    
    public int partition(int[] nums, int left, int right){
        int pivot = nums[left];
        int j = left;
        for(int i=left + 1;i <= right;i++){
            if(nums[i] < pivot){
                j++;
                swap(nums,j,i);
            }
        }
        swap(nums,i,left);
        return j;
    }
    private void swap(int[] nums,int index1, int index2){
        int temp = nums[index1];
        nums[index1] = nums[index2];
        nums[index2] = temp;
    }
}
```

### 二叉树的最近公共祖先

```java
class Solution{
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q){
        while(true){
            if(p.val > root.val && q.val > root.val){
                root = root.right;
            }else if(p.val < root.val && q.val < root.val){
                root = root.left;
            }else{
                break;
            }
        }
        return root;
    }
}
```

### 合并两个有序链表

```java
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        if(l1 == null && l2 == null) return null;
        ListNode dummy = new ListNode(0);
        ListNode cur = dummy;
        while(l1!=null&&l2 != null){
            if(l1.val < l2.val){
                cur.next = l1;
                l1 = l1.next;
            }else {
                cur.next = l2;
                l2 = l2.next;
            }
            cur = cur.next;
        }
        if(l1!=null) cur.next = l1;
        if (l2!=null) cur.next = l2;
        return dummy.next;
    }
```

### k个一组翻转链表

```java
public ListNode reverseKGroup(ListNode head, int k){
    if(head == null || head.next == null) return head;
    
    ListNode tail = head;
    for(int i=0;i<k;i++){
        if(tail == null) return head;
        tail = tail.next;
    }
    ListNode newHead = reverse(head,tail);
    head.next = reverseKGroup(tail,k);
    return newHead;
}

private reverse(ListNode head, ListNode tail){
    ListNode pre = null;
    ListNode next = null;
    while(head != tail){
        next = head.next;
        head.next = pre;
        pre = head;
        head = next;
    }
    return pre;
}
```

### 用两个栈实现队列

用两个栈实现一个队列，队列的声明如下，请实现它的两个函数appendTail和deleteHead，分别完成在队列尾部插入整数和在队列头部删除整数的功能。

```java
class CQueue{
    LinkedList<Integer> A,B;
    
    public CQueue(){
        A = new LinkedList<>();
        B = new LinkedList<>();
    }
    public void appendTail(int value){
        A.push(value);
    }
    public int deleteHead(){
        if(!B.isEmpty()) return B.pop();
        if(A.isEmpty()) return -1;
        while(!A.isEmpty()){
            B.push(A.pop());
        }
        return B.pop();
    }
}
```

### LRU缓存机制

```java
class Node {
    int key, val;
    Node prev, next;
    public Node(int key, int val){
        this.key = key;
        this.val = val;
    }    
    }
}

public calss DoubleList{
    private Node head, tail;
    private int size;
    DoubleList(){
        head = new Node(0,0);
        tail = new Node(0,0);
        head.next = tail;
        tail.prev = head;
        size = 0;
    }
    public void addFirst(Node x){
		x.prev = head;
        x.next = head.next;
        head.next.prev = x;
        head.next = x;
        size++;
    }
    
    public void remove(Node x){
		x.prev.next = x.next;
        x.next.prev = x.prev;
        size--;
    }
    
    public Node removeLast(){
        Node last = tail.prev;
        remove(last);
        return last;
    }
    
    public size(){
        return size;
    }
}

class LRU_Cache{
    HashMap<Integer, Node> map;
    int cap;
    DoubleList cache;
    
    public LRU_Cache(int capacity){
        cap = capacity;
        cache = new DoubleList();
        map = new HashMap<>();
    }
    
    public int get(int key){
        if(!map.containsKey(key))
            return -1;
        int value = map.get(key).val;
        put(key,value);
        return value;
    }
    
    public void put(int key, int value){
        Node x = new Node(key,value);
        if(map.containsKey(x)){
            cache.remove(x);
        }else{
            if(cap == cache.size()){
                Node last = cache.removeLast();
                map.remove(last.key); 	
            }
        }
        cache.addFirst(x);
        map.put(key,x);
    }
}
```

```java
public class LRUCache{
    class DLinkedNode{
        int key;
        int value;
        DLinkedNode prev;
        DLinkedNode next;
        Public DLinkedNode(){}
        public DLinkedNode(int key, int value){
            this.key = key;
            this.value = value;
        }
    }
    
    
}
```

### 二叉树的前序遍历

非递归类采用栈实现

```java
前序遍历：输出结点，压进右节点，压左节点;
public static void preOrder(TreeNode head){
    if(head == null) return;
    LinkedList<TreeNode> stack = new LinkedList<>();
    stack.push(head);
    while(!stack.isEmpty()){
        TreeNode node = stack.pop();
        sout;
        if(node.right != null){
            stack.push(node.right);
        }
        if(node.left != null){
            stack.push(node.left);
        }
    }
}
```

```java
中序遍历;
public void inOrder(TreeNode head){
    if(head == null) return;
    TreeNode cur = head;
    LinkedList<TreeNode> stack = new LinkedList<>();
    while(!stack.isEmpty() || cur != null){
        while(cur != null){
            stack.push(cur);
            cur = cur.left;
        }
        TreeNode node = stack.pop();
        sout;
        if(node.right != null) cur = node.right;
    }
}
```

```java
利用前序遍历和一个栈实现后序遍历;
public void postOrder(TreeNode head){
    if(head == null) return;
    LinkedList<TreeNode> stack1 = new LinkedList<>();
    LinkedList<TreeNode> stack2 = new LinkedList<>();
    stack1.push(head);
    while(!stack1.isEmpty()){
        TreeNode node = stack1.pop();
        stack2.push(node);
        if(node.left != null){
            stack1.push(node.left);
        }
        if(node.right != null){
            stack2.push(node.right);
        }
    }
    while(!stack2.isEmpty()){
        sout.stack2.pop();
    }
}
```

### 反转字符串

双指针

```java
class Solution {
    public void reverseString(char[] s) {
        int left = 0;
        int right = s.length - 1;
        while(left < right){
            char temp = s[left];
            s[left] = s[right];
            s[right] = temp;
            left++;
            right--;
        }
    }
}
```

### 环形链表

```java
public ListNode detectCycle(ListNode head){
    ListNode fast = head, slow = head;
    while(true){
        if(fast == null || fast.next == null) return null;
        fast = fast.next.next;
        slow = slow.next;
        if(fast == slow) break;
    }
    fast = head;
    while(fast != slow){
        fast = fast.next;
        slow = slow.next;
    }
    return fast;
}
```

### 二叉树的最大深度

```java
public int maxDepth(TreeNode root){
    if(root == null) return 0;
    return Math.max(maxDepth(root.left),maxDepth(root.right)) + 1;
}
```

### 字符串解码

```java
class Solution{
    public String decodeString(String s){
        StringBuilder res = new StringBuilder();
        int multi = 0;
        LinkedList<Integer> stack_multi = new LinkedList<>();
        LinkedList<String> stack_res = new LinkedList<>();
        for(Character c : s.toCharArray()){
            if(c == '['){
                stack_multi.push(multi);
                stack_res.push(res.toString());
                res = new StringBuilder();
                multi = 0;
            }else if(c == ']'){
                StringBuilder temp = new StringBuilder();
                int multi1 = stack_multi.pop();
                for(int i=0;i<multi1;i++){
                    temp.append(res);
                }
                res = new StringBuilder(stack_res.pop() + temp);
            }else if(c >= '0' && c <= '9'){
                multi = multi * 10 + Integer.parseInt(c + "");
            }else res.append(c);
        }
        return res.toString();
    }
}
```

### 无重复最长字符串

```java
class Solution{
    public int lengthOfLongestSubstring(String s){
        HashSet<Integer> set = new HashSet<>();
        int ans = 0, rk=0;
        int n = s.length();
        for(int i = 0;i < n;i++){
            if(i!=0){
                set.remove(s.charAt(i-1));
            }
            while(rk < n&&!set.containsKey(rk)){
                set.add(s.charAt(rk));
                rk++;
            }
            ans = Math.max(ans, rk - i);
        }
        return ans;
    }
}
```

### 两数之和

```java
class Solution{
    public int[] twoSum(int[] nums){
        HashMap<Integer, Integer> map = new HashMap<>();
        int[] res = new int[2];
        
        for(int i=0;i<nums.length;i++){
            int dif = target - nums[i];
            if(map.get(dif) != null){
                res[0] = map.get(dif);
                res[1] = map.get(nums[i]);
                return res;
            }
            map.put(nums[i],i);
        }
        return res;
    }
}
```

### 二叉树的中序遍历

递归

### 搜索旋转排序数组

```java
class Solution{
    public int search(int nums[], int target){
        int n = nums.length;
        if(nums == null) return -1;
        if(n == 1) return nums[0] == target ? 0:-1;
        int i = 0, j = n - 1;
        while(i <= j){
            int mid = (i + j) / 2;
            if(nums[mid] == target) return mid;
            if(nums[0] <= nums[mid]){
                if(nums[0] <= target && target < nums[mid]){
                    j = mid - 1;
                }else{
                    i = mid + 1;
                }
            }else{
                if(nums[mid] < target && target <= nums[n-1]){
                    i = mid + 1;
                }else{
                    j = mid - 1;
                }
            }
        }
        return -1;
    }
}
```

### 颠倒二进制位

```java
public int reverseBits(int n){
    int res = 0;
    for(int i=0;i<32;i++){
        int cur = n&1;
        res = res + (cur << (31-i));
        n >> 1;
    }
    return res;
}
```

### 旋转图像

**思路**：先对角交换，在对折

```java
public void rotate(int[][] matrix){
    int n = matrix.length;
    for(int i=0;i<n;i++){
        for(int j=i;j<n;j++){
            int tmp = matrix[i][j];
            matrix[i][j] = matrix[j][i];
            matrix[j][i] = tmp;
        }
    }
    for(int i=0;i<n;i++){
        for(int j=0;j<n / 2;j++){
            int tmp = matrix[i][j];
            matrix[i][j] = matrix[i][n-j-1];
            matrix[i][n-j-1] = tmp;
        }
    }
}
```

### 生成者消费者模型

```java
package cn.thread;

import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingQueue;

/**
 * java多线程模拟生产者消费者问题
 * 
 * ProducerConsumer是主类，Producer生产者，Consumer消费者，Product产品，Storage仓库
 * 
 * @author 林计钦
 * @version 1.0 2013-7-24 下午04:49:02
 */
public class ProducerConsumer {
    public static void main(String[] args) {
        ProducerConsumer pc = new ProducerConsumer();

        Storage s = pc.new Storage();

        ExecutorService service = Executors.newCachedThreadPool();
        Producer p = pc.new Producer("张三", s);
        Producer p2 = pc.new Producer("李四", s);
        Consumer c = pc.new Consumer("王五", s);
        Consumer c2 = pc.new Consumer("老刘", s);
        Consumer c3 = pc.new Consumer("老林", s);
        service.submit(p);
        //service.submit(p2);
        service.submit(c);
        service.submit(c2);
        service.submit(c3);
        
    }

    /**
     * 消费者
     * 
     * @author 林计钦
     * @version 1.0 2013-7-24 下午04:53:30
     */
    class Consumer implements Runnable {
        private String name;
        private Storage s = null;

        public Consumer(String name, Storage s) {
            this.name = name;
            this.s = s;
        }

        public void run() {
            try {
                while (true) {
                    System.out.println(name + "准备消费产品.");
                    Product product = s.pop();
                    System.out.println(name + "已消费(" + product.toString() + ").");
                    System.out.println("===============");
                    Thread.sleep(500);
                }
            } catch (InterruptedException e) {
                e.printStackTrace();
            }

        }

    }

    /**
     * 生产者
     * 
     * @author 林计钦
     * @version 1.0 2013-7-24 下午04:53:44
     */
    class Producer implements Runnable {
        private String name;
        private Storage s = null;

        public Producer(String name, Storage s) {
            this.name = name;
            this.s = s;
        }

        public void run() {
            try {
                while (true) {
                    Product product = new Product((int) (Math.random() * 10000)); // 产生0~9999随机整数
                    System.out.println(name + "准备生产(" + product.toString() + ").");
                    s.push(product);
                    System.out.println(name + "已生产(" + product.toString() + ").");
                    System.out.println("===============");
                    Thread.sleep(500);
                }
            } catch (InterruptedException e1) {
                e1.printStackTrace();
            }

        }
    }

    /**
     * 仓库，用来存放产品
     * 
     * @author 林计钦
     * @version 1.0 2013-7-24 下午04:54:16
     */
    public class Storage {
        BlockingQueue<Product> queues = new LinkedBlockingQueue<Product>(10);

        /**
         * 生产
         * 
         * @param p
         *            产品
         * @throws InterruptedException
         */
        public void push(Product p) throws InterruptedException {
            queues.put(p);
        }

        /**
         * 消费
         * 
         * @return 产品
         * @throws InterruptedException
         */
        public Product pop() throws InterruptedException {
            return queues.take();
        }
    }

    /**
     * 产品
     * 
     * @author 林计钦
     * @version 1.0 2013-7-24 下午04:54:04
     */
    public class Product {
        private int id;

        public Product(int id) {
            this.id = id;
        }

        public String toString() {// 重写toString方法
            return "产品：" + this.id;
        }
    }

}
```

```json
# json格式
{
    "emplyees": [
        {"firstName":"Bill", "lastName":"Gates"},
        {"fistName":"George", "lastName":"Bush"},
        {"firstName":"Thomas","lastName":"Carter"}
    ]
}

//2
{
    "id": 1,
    "name": "Java核心技术",
    "author": {
        "firstName": "Abc",
        "lastName": "Xyz"
    },
    "isbn": "1234567",
    "tags": ["Java", "Network"]
}
```

### 圆圈中最后剩下的数字

```java

```

### n个色子的概率

```java
public double[] twoSum(int n){
    double pre[] = {1/6d,1/6d,1/6d,1/6d,1/6d,1/6d};
    for(int i=2;i<=n;i++){
        double tmp[] = new double[5*i+1];
        for(int j=0;j<pre.length;j++){
            for(int x=0;x<6;x++){
                tmp[j+x] += pre[j]/6;
            }
        }
        pre = tmp;
    }
    return pre;
}
```

```java
//死锁
public void add(int m){
    synchronized(lockA){
        this.value += m;
        sychronoized(lockB){
            this.another += m;
        }
    }
}

public void dec(int m){
    synchronized(lockB){
        this.another -= m;
        synchronized(lockA){
            this.value -= m;
        }
    }
}
```

### 滑动窗口的最大值

**思路**：双端队列实现

- `deque`内仅包含窗口内的元素=>每滑动窗口，需将deque中的头元素删除
- `deque`内的元素非严格递减=>每轮窗口滑动添加了元素`nums[j+1]`，需将`deque`内所有`<nums[j+1]`的元素删除

```java
class Solution{
    public int[] maxSlidingWindow(int[] nums, int k){
        if(nums.length == 0 || k==0) return new int[0];
        Deque<Integer> deque = new LinkedList<>();
        int[] res = new int[nums.length - k + 1];
        for(int j=0,i=1-k;j<nums.length;i++,j++){
            if(i>0 && nums[i-1] == deque.peekFirst()) deque.removeFirst();
            while(!deque.isEmpty()&&deque.peekLast() < nums[j])deque.removeLast();
            deque.addLast(nums[j]);
            if(i>=0) res[i] = deque.peekFirst();
        }
        return res;
    }
}
```

### 重建二叉树

前序和中序遍历：思路：根据前序和中序遍历的顺序，可以知道前序遍历的第一个结点是根结点，通过在中序遍历找到根结点，可以把树分成左子树和右子树，之后可以采用递归的方式实现重建。需要注意的是每次前序遍历的头结点，因为用它来确定根结点的。

```java
class Solution{
    HashMap<Integer, Integer> dict = new HashMap<>();
    int[] po;
    public TreeNode buildTree(int[] preOrder, int[] inOrder){
        po = preOrder;
        for(int i=0;i<inOrder.length;i++){
            dict.put(inOrder[i],i);
        }
        return recur(0,0,inOrder.length-1);
    }
    
    public TreeNode recur(int left_root, int in_left, int in_right){
        if(in_left > in_right) return null;
        TreeNode root = new TreeNode(po[left_root]);
        int i = dict.get(po[left_root]);
        root.left = recur(left_root + 1, in_left, i-1);
        root.right = recur(left_root + i - in_left + 1, i+1,in_right);
        return root;
    }
}
```

### 双重检验锁实现单例模式

```java
class Singleton{
    //静态实例变量
    private volatile static Singleton uniqueInstance;
    //私有化构造函数
    private Singleton(){}
    
    public static Singleton getUniqueInstance(){
        if(uniqueInstance == null){//第一重判断
            synchronized(Singleton.class){
                if(uniqueInstance == null){//第二重判断
                    uniqueInstance = new Singleton();
                }
            }
        }
        return uniqueInstance;
    }
}
```

```java
/**
 * 双重检验锁方式实现单例模式
 */
public class DualLazySingleTon {
	// 静态实例变量
	private volatile static DualLazySingleTon instance;
 
	// 私有化构造函数
	private DualLazySingleTon() {
		System.out.println(Thread.currentThread().getName() + "\t" + "进入构造方法");
	}
 
	// 静态public方法，向整个应用提供单例获取方式
	public static DualLazySingleTon getInstance() {
		if (instance == null) { //第1重判断
			synchronized (DualLazySingleTon.class) {
				if (instance == null) { //第2重判断
					instance = new DualLazySingleTon();	
				}
			}
		}
		return instance;
	}
 
}
```

### 生产者消费者模型-synchronized实现

```java
import java.util.LinkedList;

public class Main {
    //常量
    private static int MAX_VALUE = 100;
    public static void main(String[] args) {
        Concurrentcomm con = new Concurrentcomm();
        new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    for (int i= 0;i < MAX_VALUE;i++){
                        Thread.sleep(0);
                        con.product();
                    }
                }catch (Exception e){
                    e.printStackTrace();
                }
            }
        }).start();
        //消费者
        new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    Thread.sleep(0);
                    for (int i=0;i<MAX_VALUE;i++){
                        con.customer();
                    }
                }catch (Exception e){
                    e.printStackTrace();
                }
            }
        }).start();
    }
}

class Concurrentcomm{
    //常量
    private static int MAX_VALUE = 10;
    //缓存
    LinkedList<String> linkedList = new LinkedList<>();
    Object object = new Object();

    /**
     * 生产者方法
     * @throws Exception
     */
    public void product() throws Exception{
        synchronized (linkedList){
            while (MAX_VALUE == linkedList.size()){
                System.out.println("仓库已满,【生产者】：暂时不能执行生成任务！");
                linkedList.wait();
            }
            linkedList.push("李四");
            System.out.println("【生产者】：生产了一个产品\t【现仓储量为】：" + linkedList.size());
            linkedList.notify();
        }
    }
    public void customer() throws Exception{
        /**
         * 根据jdk的void notifyAll()的描述，“解除那些在该对象上调用wait()方法的线程的阻塞状态。该方法只能在同步方法或同步块内部调用。
         * 如果当前线程不是对象所得持有者，
         * 该方法抛出一个java.lang.IllegalMonitorStateException异常
         * 因此，我们使用同一把锁
         */
        synchronized (linkedList){
            //多线程判断中使用while不要使用if否则会出现虚假唤醒问题
            while(linkedList.size() == 0){
                System.out.println("仓库无货，【消费者】：暂时不能执行消费任务！");
                linkedList.wait();
            }
            linkedList.poll();
            System.out.println("【消费者】：消费了一个产品\t【现仓储量为】：" + linkedList.size());
            linkedList.notifyAll();
        }
    }
}
```

### Java对HashMap中的key或者value值进行排序

```java
import java.util.*;

public class Main {
    public static void main(String[] args) {
        Map<String, Integer> map = new HashMap<>();
        map.put("d",2);
        map.put("c",1);
        map.put("b",4);
        map.put("a",3);

        List<Map.Entry<String,Integer>> infoIds = new ArrayList<Map.Entry<String, Integer>>(map.entrySet());

        Collections.sort(infoIds, new Comparator<Map.Entry<String, Integer>>() {
            @Override
            public int compare(Map.Entry<String, Integer> o1, Map.Entry<String, Integer> o2) {
                return (o1.getKey()).toString().compareTo(o2.getKey().toString());
            }
        });

        for (int i=0;i<infoIds.size();i++){
            String id = infoIds.get(i).toString();
            System.out.println(id + "");
        }
        System.out.println();

        Collections.sort(infoIds, new Comparator<Map.Entry<String, Integer>>() {
            @Override
            public int compare(Map.Entry<String, Integer> o1, Map.Entry<String, Integer> o2) {
                return (o1.getValue().toString()).compareTo(o2.getValue().toString());
            }
        });

        for (int i=0;i<infoIds.size();i++){
            String id = infoIds.get(i).toString();
            System.out.println(id + " ");
        }
    }
}
```

### 数组中第k个最大元素

快速排序

```java
class Solution{
    private static Random random = new Random(System.currentTimeMills());
    public int findKthLargest(int[] nums,int k){
        int len = nums.length;
        int left = 0;
        int right = len - 1;
        
        int target = len - k;
        
        while(true){
            int index = partition(nums,left,right);
            if(index == target){
                return nums[index];
            }else if(index < target){
                left = index + 1;
            }else{
                right = index - 1;
            }
        }
    }
    
    public int partition(int[] nums,int left, int right){
        if(right > left){
            int randomIndex = left + 1 + random.nextInt(right - left);
            swap(nums,left,randomIndex);
        }
        int pivot = nums[left];
        
        int lt = left + 1;
        int rt = right;
        
        while(true){
            while(lt <= rt && nums[lt] < pivot){
                lt++;
            }
            while(lt <= rt && nums[rt] > pivot){
                rt--;
            }
            if(lt > rt){
                break;
            }
            swap(nums,lt,rt);
            lt++;
            rt--;
        }
        swap(nums,left,rt);
        return rt;
    }
    
    public void swap(int[] nums,int left, int right){
        int tmp = nums[left];
        nums[left] = nums[right];
        nums[right] = tmp;
    }
}
```

### 合并k个链表

#### 堆方法：

建立一个空间为k的小顶堆，把每个链表的首元素放入堆中，每次取出堆顶元素。

```java
class Solution{
    public ListNode mergeKLists(ListNode[] lists){
        if(lists == null || lists.length == 0) return null;
        PriorityQueue<ListNode> pq = new PriorityQueue<>(new Comparator<ListNode>(){
            public int compare(ListNode o1,ListNode o2){
                return (o1.val - o2.va1);
            }
        });
        //哑结点
        ListNode dummy = new ListNode(-1);
        ListNode cur = dummy;
        for(int i=0;i<lists.length;i++){
            ListNode head = lists[i];
            if(head != null){
                pq.add(head);
            }
        }
        while(queue.size() > 0){
            ListNode node = pq.poll();
            cur.next = node;
            cur = cur.next;
            if(node.next != null){
                pq.add(node.next);
            }
        }
        cur.next = null;
        return dummy.next;
    }
}
```







